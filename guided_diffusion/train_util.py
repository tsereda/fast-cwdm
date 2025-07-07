import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp
import wandb

import itertools
import numpy as np

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    if _max > _min:
        normalized_img = (img - _min) / (_max - _min)
    else:
        normalized_img = np.zeros_like(img)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        contr,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        summary_writer=None,
        mode='default',
        loss_level='image',
        use_fast_ddpm=False,          # NEW: Enable Fast-DDPM
        fast_ddpm_strategy='non-uniform',  # NEW: Fast-DDPM sampling strategy
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.contr = contr
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        else:
            self.grad_scaler = amp.GradScaler(enabled=False)
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')
        self.loss_level = loss_level
        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()
        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )
        # Fast-DDPM integration
        self.use_fast_ddpm = use_fast_ddpm
        self.fast_ddpm_strategy = fast_ddpm_strategy
        if self.use_fast_ddpm:
            from .gaussian_diffusion import FastDDPMScheduleSampler
            self.fast_ddpm_sampler = FastDDPMScheduleSampler(
                num_timesteps=diffusion.num_timesteps,
                strategy=fast_ddpm_strategy
            )
            print(f"[TRAIN] Using Fast-DDPM with {fast_ddpm_strategy} sampling")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        t = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
            if self.dataset in ['brats']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}

            if self.mode=='i2i':
                batch['t1n'] = batch['t1n'].to(dist_util.dev())
                batch['t1c'] = batch['t1c'].to(dist_util.dev())
                batch['t2w'] = batch['t2w'].to(dist_util.dev())
                batch['t2f'] = batch['t2f'].to(dist_util.dev())
            else:
                batch = batch.to(dist_util.dev())

            t_fwd = time.time()
            t_load = t_fwd-t

            lossmse, sample, sample_idwt = self.run_step(batch, cond)

            t_fwd = time.time()-t_fwd

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

            # TensorBoard logging
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', lossmse.item(), global_step=self.step + self.resume_step)

            # wandb logging
            wandb_log_dict = {
                'time/load': t_load,
                'time/forward': t_fwd,
                'time/total': t_total,
                'loss/MSE': lossmse.item(),
                'step': self.step + self.resume_step
            }

            if self.step % 200 == 0:
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                              global_step=self.step + self.resume_step)
                # wandb image logging
                img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                wandb_log_dict['sample/x_0'] = wandb.Image(img, caption='sample/x_0')

                image_size = sample.size()[2]
                for ch in range(8):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)
                    img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                    wandb_log_dict[f'sample/{names[ch]}'] = wandb.Image(img, caption=f'sample/{names[ch]}')

                if self.mode == 'i2i':
                    if not self.contr == 't1n':
                        image_size = batch['t1n'].size()[2]
                        midplane = batch['t1n'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t1n', midplane.unsqueeze(0),
                                                      global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t1n'] = wandb.Image(img, caption='source/t1n')
                    if not self.contr == 't1c':
                        image_size = batch['t1c'].size()[2]
                        midplane = batch['t1c'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t1c', midplane.unsqueeze(0),
                                                      global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t1c'] = wandb.Image(img, caption='source/t1c')
                    if not self.contr == 't2w':
                        midplane = batch['t2w'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t2w', midplane.unsqueeze(0),
                                                      global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t2w'] = wandb.Image(img, caption='source/t2w')
                    if not self.contr == 't2f':
                        midplane = batch['t2f'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t2f', midplane.unsqueeze(0),
                                                      global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t2f'] = wandb.Image(img, caption='source/t2f')

            # Actually log to wandb
            wandb.log(wandb_log_dict, step=self.step + self.resume_step)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with torch.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not torch.isfinite(lossmse): #infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                logger.log(f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None

        if self.mode == 'i2i':
            batch_size = batch['t1n'].shape[0]
        else:
            batch_size = batch.shape[0]
        # MODIFIED: Use Fast-DDPM timestep sampling if enabled
        if self.use_fast_ddpm:
            t, weights = self.fast_ddpm_sampler.sample(batch_size, dist_util.dev())
            print(f"[TRAIN STEP] Fast-DDPM sampled timesteps: {t.cpu().numpy()}")
        else:
            t, weights = self.schedule_sampler.sample(batch_size, dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            x_start=batch,
            t=t,
            model_kwargs=cond,
            labels=label,
            mode=self.mode,
            contr=self.contr
        )
        losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses1["loss"].detach())

        losses = losses1[0]         # Loss value
        sample = losses1[1]         # Denoised subbands at t=0
        sample_idwt = losses1[2]    # Inverse wavelet transformed denoised subbands at t=0

        # Log wavelet level loss
        self.summary_writer.add_scalar('loss/mse_wav_lll', losses["mse_wav"][0].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_llh', losses["mse_wav"][1].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_lhl', losses["mse_wav"][2].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_lhh', losses["mse_wav"][3].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_hll', losses["mse_wav"][4].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_hlh', losses["mse_wav"][5].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_hhl', losses["mse_wav"][6].item(),
                                       global_step=self.step + self.resume_step)
        self.summary_writer.add_scalar('loss/mse_wav_hhh', losses["mse_wav"][7].item(),
                                       global_step=self.step + self.resume_step)

        weights = th.ones(len(losses["mse_wav"])).cuda()  # Equally weight all wavelet channel losses

        loss = (losses["mse_wav"] * weights).mean()
        lossmse = loss.detach()

        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        # perform some finiteness checks
        if not torch.isfinite(loss):
            logger.log(f"Encountered non-finite loss {loss}")
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        return lossmse.detach(), sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log("Saving model...")
                if self.dataset == 'brats':
                    filename = f"brats_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'brats_inpainting':
                    filename = f"brats_inpainting_{(self.step + self.resume_step):06d}.pt"
                elif self.dataset == 'synthrad':
                    filename = f"synthrad_{(self.step + self.resume_step):06d}.pt"
                else:
                    raise ValueError(f'dataset {self.dataset} not implemented')

                with bf.BlobFile(bf.join(get_blob_logdir(), 'checkpoints', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())

        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            with bf.BlobFile(
                bf.join(checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
