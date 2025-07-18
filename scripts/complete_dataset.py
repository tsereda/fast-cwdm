#!/usr/bin/env python3
"""
Simple medical image synthesis script - fixes tensor shape issues
"""

import argparse
import nibabel as nib
import numpy as np
import os
import torch as th
import glob
import sys

sys.path.append(".")

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults, 
    create_model_and_diffusion,
    args_to_dict
)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

# Constants
MODALITIES = ['t1n', 't1c', 't2w', 't2f']


def load_image(file_path):
    """Load and preprocess a single image."""
    print(f"Loading: {file_path}")
    
    # Load image
    img = nib.load(file_path).get_fdata()
    print(f"  Original shape: {img.shape}")
    
    # Normalize
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))
    
    # Convert to tensor and preprocess like training
    img_tensor = th.zeros(1, 240, 240, 160)
    img_tensor[:, :, :, :155] = th.tensor(img_normalized)
    
    # CRITICAL FIX: Crop to 224x224x155 (not 160!)
    img_tensor = img_tensor[:, 8:-8, 8:-8, :155]
    
    print(f"  Preprocessed shape: {img_tensor.shape}")
    return img_tensor.float()


def get_case_prefix(case_dir):
    """Detect the file prefix for the case by inspecting the filenames in the directory."""
    for filename in os.listdir(case_dir):
        if filename.endswith('.nii.gz'):
            # Remove modality and extension
            for modality in MODALITIES:
                if filename.endswith(f"-{modality}.nii.gz"):
                    return filename[:-(len(f"-{modality}.nii.gz"))]
    return None

def find_missing_modality(case_dir):
    """Find which modality is missing, using detected file prefix."""
    case_prefix = get_case_prefix(case_dir)
    if not case_prefix:
        print(f"Could not determine case prefix in {case_dir}")
        return None
    for modality in MODALITIES:
        file_path = os.path.join(case_dir, f"{case_prefix}-{modality}.nii.gz")
        if not os.path.exists(file_path):
            return modality
    return None


def load_available_modalities(case_dir, missing_modality):
    """Load all available modalities using detected file prefix."""
    case_prefix = get_case_prefix(case_dir)
    if not case_prefix:
        raise RuntimeError(f"Could not determine case prefix in {case_dir}")
    available = [m for m in MODALITIES if m != missing_modality]
    modalities = {}
    for modality in available:
        file_path = os.path.join(case_dir, f"{case_prefix}-{modality}.nii.gz")
        modalities[modality] = load_image(file_path)
    return modalities


def find_checkpoint(missing_modality, checkpoint_dir):
    """Find the best checkpoint for the missing modality."""
    # Look for BEST checkpoints first
    pattern = f"brats_{missing_modality}_BEST_*.pt"
    best_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if best_files:
        checkpoint = best_files[0]
        print(f"Found BEST checkpoint: {checkpoint}")
        return checkpoint
    
    # Fallback to regular checkpoints
    pattern = f"brats_{missing_modality}_*.pt"
    regular_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not regular_files:
        raise FileNotFoundError(f"No checkpoint found for {missing_modality}")
    
    # Sort by iteration number
    def get_iteration(filename):
        parts = os.path.basename(filename).split('_')
        try:
            return int(parts[2])
        except (IndexError, ValueError):
            return 0
    
    regular_files.sort(key=get_iteration, reverse=True)
    checkpoint = regular_files[0]
    print(f"Found checkpoint: {checkpoint}")
    return checkpoint


def parse_checkpoint_info(checkpoint_path):
    """Parse checkpoint filename to get training parameters."""
    basename = os.path.basename(checkpoint_path)
    
    # Default values
    diffusion_steps = 1000
    sample_schedule = "direct"
    
    # Parse filename: brats_t1n_BEST_sampled_10.pt
    if "_BEST_" in basename:
        parts = basename.split('_')
        if len(parts) >= 4:
            sample_schedule = parts[3]
        if len(parts) >= 5:
            try:
                diffusion_steps = int(parts[4].split('.')[0])
            except ValueError:
                pass
    
    print(f"Checkpoint config: schedule={sample_schedule}, steps={diffusion_steps}")
    return sample_schedule, diffusion_steps


def create_model_args(sample_schedule="direct", diffusion_steps=1000):
    """Create model arguments."""
    class Args:
        pass
    
    args = Args()
    
    # Model architecture
    args.image_size = 224
    args.num_channels = 64
    args.num_res_blocks = 2
    args.channel_mult = "1,2,2,4,4"
    args.learn_sigma = False
    args.class_cond = False
    args.use_checkpoint = False
    args.attention_resolutions = ""
    args.num_heads = 1
    args.num_head_channels = -1
    args.num_heads_upsample = -1
    args.use_scale_shift_norm = False
    args.dropout = 0.0
    args.resblock_updown = True
    args.use_fp16 = False
    args.use_new_attention_order = False
    args.dims = 3
    args.num_groups = 32
    args.bottleneck_attention = False
    args.resample_2d = False
    args.additive_skips = False
    args.use_freq = False
    
    # Diffusion parameters
    args.predict_xstart = True
    args.noise_schedule = "linear"
    args.timestep_respacing = ""
    args.use_kl = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    
    # Model channels: 8 (target) + 24 (3 modalities * 8 DWT components each)
    args.in_channels = 32
    args.out_channels = 8
    
    # From checkpoint
    args.diffusion_steps = diffusion_steps
    args.sample_schedule = sample_schedule
    args.mode = 'i2i'
    args.dataset = "brats"
    
    return args


def prepare_conditioning(available_modalities, missing_modality, device):
    """Prepare conditioning tensor from available modalities."""
    dwt = DWT_3D("haar")
    
    # Get modalities in consistent order
    available_order = [m for m in MODALITIES if m != missing_modality]
    print(f"Available modalities: {available_order}")
    
    cond_list = []
    
    for modality in available_order:
        # Get tensor and add channel dimension
        tensor = available_modalities[modality].to(device)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)  # [B, 1, D, H, W]
        
        print(f"  {modality} input shape: {tensor.shape}")
        
        # Apply DWT
        dwt_components = dwt(tensor)
        shapes = [c.shape for c in dwt_components]
        print(f"  {modality} DWT shapes: {shapes}")
        
        # Find minimum z-dimension to fix mismatches
        min_z = min(c.shape[-1] for c in dwt_components)
        print(f"  {modality} min z-dimension: {min_z}")
        
        # Crop all components to same size
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt_components
        cropped_components = [
            LLL[:, :, :, :, :min_z] / 3.,  # Divide LLL by 3 as per training
            LLH[:, :, :, :, :min_z],
            LHL[:, :, :, :, :min_z],
            LHH[:, :, :, :, :min_z],
            HLL[:, :, :, :, :min_z],
            HLH[:, :, :, :, :min_z],
            HHL[:, :, :, :, :min_z],
            HHH[:, :, :, :, :min_z]
        ]
        
        # Concatenate DWT components
        modality_cond = th.cat(cropped_components, dim=1)
        print(f"  {modality} final conditioning: {modality_cond.shape}")
        cond_list.append(modality_cond)
    
    # Concatenate all modalities
    cond = th.cat(cond_list, dim=1)
    print(f"Final conditioning shape: {cond.shape}")
    
    return cond
    dwt = DWT_3D("haar")

    # Get modalities in consistent order
    available_order = [m for m in MODALITIES if m != missing_modality]
    print(f"Available modalities: {available_order}")

    cond_list = []
    dwt_shapes = []
    dwt_components_all = []

    # First pass: collect DWT shapes for all modalities
    for modality in available_order:
        tensor = available_modalities[modality].to(device)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)
        print(f"  {modality} input shape: {tensor.shape}")
        dwt_components = dwt(tensor)
        dwt_components_all.append(dwt_components)
        dwt_shapes.append([c.shape for c in dwt_components])
        print(f"  {modality} DWT shapes: {[c.shape for c in dwt_components]}")

    # Find minimum shape for each DWT component across all modalities
    min_shapes = []
    for i in range(8):
        min_shape = list(dwt_components_all[0][i].shape)
        for comps in dwt_components_all:
            for d in range(2, len(min_shape)):
                min_shape[d] = min(min_shape[d], comps[i].shape[d])
        min_shapes.append(tuple(min_shape))

    # Second pass: crop all DWT components to min_shapes and stack
    for m_idx, modality in enumerate(available_order):
        cropped = []
        for i, comp in enumerate(dwt_components_all[m_idx]):
            target_shape = min_shapes[i]
            slices = [slice(0, s) for s in target_shape]
            cropped_comp = comp[tuple(slices)]
            cropped.append(cropped_comp)
        # Concatenate along channel dim (should be 8)
        cond = th.cat(cropped, dim=1)
        print(f"  {modality} min z-dimension: {min([c.shape[-1] for c in cropped])}")
        print(f"  {modality} final conditioning: {cond.shape}")
        cond_list.append(cond)

    # Concatenate all modalities along channel dim
    conditioning = th.cat(cond_list, dim=1)
    print(f"Final conditioning shape: {conditioning.shape}")
    return conditioning


def synthesize_modality(available_modalities, missing_modality, checkpoint_path, device):
    # --- Fast-DDPM inference workaround: patch beta schedule for 10-step Fast-DDPM ---
    import numpy as np
    import guided_diffusion.gaussian_diffusion as gd
    def patch_beta_schedule_for_fast_ddpm(diffusion_steps, sample_schedule):
        orig_get_named_beta_schedule = gd.get_named_beta_schedule
        def patched_get_named_beta_schedule(schedule_name, num_diffusion_timesteps, sample_schedule_arg="direct"):
            if num_diffusion_timesteps == 10 and sample_schedule == "sampled":
                print("[PATCH] Fast-DDPM detected: using indices 0..9 for beta schedule")
                indices = np.arange(10)
                betas_1000 = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
                betas = betas_1000[indices]
                print(f"[PATCH] Sampled betas range: {betas[0]:.6f} → {betas[-1]:.6f}")
                return betas
            else:
                return orig_get_named_beta_schedule(schedule_name, num_diffusion_timesteps, sample_schedule_arg)
        gd.get_named_beta_schedule = patched_get_named_beta_schedule
    # Patch before model/diffusion creation
    sample_schedule, diffusion_steps = parse_checkpoint_info(checkpoint_path)
    patch_beta_schedule_for_fast_ddpm(diffusion_steps, sample_schedule)
    """Synthesize the missing modality."""
    print(f"\n=== Synthesizing {missing_modality} ===")
    
    # --- Fast-DDPM fix: always use the number of steps the model was trained with ---
    # For your checkpoints (e.g., brats_t1n_BEST_sampled_10.pt), the model was trained with 10 steps
    model_steps = diffusion_steps  # This is 10 for your Fast-DDPM checkpoints
    args = create_model_args(sample_schedule, model_steps)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    
    # Load weights
    print(f"Loading model from: {checkpoint_path}")
    state_dict = dist_util.load_state_dict(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Print model device and dtype
    first_param = next(model.parameters())
    print(f"Model device: {first_param.device}, dtype: {first_param.dtype}")

    # Prepare conditioning
    cond = prepare_conditioning(available_modalities, missing_modality, device)

    # Check cond device, dtype, NaN/Inf
    print(f"Cond device: {cond.device}, dtype: {cond.dtype}")
    print(f"Cond has NaN: {th.isnan(cond).any().item()}, Inf: {th.isinf(cond).any().item()}")

    # Create noise tensor with matching dimensions
    _, _, cond_d, cond_h, cond_w = cond.shape
    noise_shape = (1, 8, cond_d, cond_h, cond_w)
    noise = th.randn(*noise_shape, device=device)

    # Check noise device, dtype, NaN/Inf
    print(f"Noise device: {noise.device}, dtype: {noise.dtype}")
    print(f"Noise has NaN: {th.isnan(noise).any().item()}, Inf: {th.isinf(noise).any().item()}")

    print(f"Noise shape: {noise.shape}")
    print(f"Conditioning shape: {cond.shape}")

    # Sample
    print(f"Running {diffusion.num_timesteps}-step sampling...")
    with th.no_grad():
        sample = diffusion.p_sample_loop(
            model=model,
            shape=noise.shape,
            noise=noise,
            cond=cond,
            clip_denoised=True,
            model_kwargs={}
        )

    print(f"Sample shape: {sample.shape}")
    
    # Convert back to spatial domain using IDWT
    idwt = IDWT_3D("haar")
    B, _, D, H, W = sample.shape
    
    spatial_sample = idwt(
        sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,  # Multiply LLL by 3
        sample[:, 1, :, :, :].view(B, 1, D, H, W),
        sample[:, 2, :, :, :].view(B, 1, D, H, W),
        sample[:, 3, :, :, :].view(B, 1, D, H, W),
        sample[:, 4, :, :, :].view(B, 1, D, H, W),
        sample[:, 5, :, :, :].view(B, 1, D, H, W),
        sample[:, 6, :, :, :].view(B, 1, D, H, W),
        sample[:, 7, :, :, :].view(B, 1, D, H, W)
    )
    
    print(f"Spatial sample shape: {spatial_sample.shape}")
    
    # Post-process
    spatial_sample = th.clamp(spatial_sample, 0, 1)
    
    # Apply brain mask from first available modality
    first_modality = list(available_modalities.values())[0].to(device)
    if first_modality.dim() == 4:
        first_modality = first_modality.unsqueeze(1)
    
    spatial_sample[first_modality == 0] = 0
    
    # Remove batch and channel dimensions
    if spatial_sample.dim() == 5:
        spatial_sample = spatial_sample.squeeze(1)  # Remove channel
    spatial_sample = spatial_sample[0]  # Remove batch
    
    print(f"Final output shape: {spatial_sample.shape}")
    return spatial_sample


def save_result(synthesized, case_dir, missing_modality, output_dir):
    """Save the synthesized modality using detected file prefix."""
    case_prefix = get_case_prefix(case_dir)
    case_name = os.path.basename(case_dir)
    # Create output directory
    output_case_dir = os.path.join(output_dir, case_name)
    os.makedirs(output_case_dir, exist_ok=True)
    # Copy existing files
    for filename in os.listdir(case_dir):
        if filename.endswith('.nii.gz'):
            src = os.path.join(case_dir, filename)
            dst = os.path.join(output_case_dir, filename)
            nib.save(nib.load(src), dst)
    # Save synthesized modality
    output_path = os.path.join(output_case_dir, f"{case_prefix}-{missing_modality}.nii.gz")
    # Get reference for header/affine
    reference_files = [f for f in os.listdir(case_dir) 
                      if f.endswith('.nii.gz') and any(m in f for m in MODALITIES)]
    if reference_files:
        reference_img = nib.load(os.path.join(case_dir, reference_files[0]))
        # Convert to numpy and pad back to original size
        synthesized_np = synthesized.detach().cpu().numpy()
        # Pad back to 240x240x155 (reverse the 8-pixel crop)
        padded = np.zeros((240, 240, 155))
        padded[8:232, 8:232, :synthesized_np.shape[2]] = synthesized_np
        # Create NIfTI image
        synthesized_img = nib.Nifti1Image(padded, reference_img.affine, reference_img.header)
    else:
        synthesized_np = synthesized.detach().cpu().numpy()
        synthesized_img = nib.Nifti1Image(synthesized_np, np.eye(4))
    nib.save(synthesized_img, output_path)
    print(f"✅ Saved: {output_path}")


def process_case(case_dir, output_dir, checkpoint_dir, device):
    """Process a single case."""
    case_name = os.path.basename(case_dir)
    print(f"\n=== Processing {case_name} ===")
    
    # Find missing modality
    missing_modality = find_missing_modality(case_dir)
    if not missing_modality:
        print(f"No missing modality in {case_name}")
        return False
    
    print(f"Missing modality: {missing_modality}")
    
    try:
        # Load available modalities
        available_modalities = load_available_modalities(case_dir, missing_modality)
        
        # Find checkpoint
        checkpoint_path = find_checkpoint(missing_modality, checkpoint_dir)
        
        # Synthesize
        synthesized = synthesize_modality(available_modalities, missing_modality, checkpoint_path, device)
        
        # Save result
        save_result(synthesized, case_dir, missing_modality, output_dir)
        
        print(f"✅ Successfully processed {case_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Simple medical image synthesis")
    parser.add_argument("--input_dir", default="./datasets/BRATS2023/pseudo_validation")
    parser.add_argument("--output_dir", default="./datasets/BRATS2023/pseudo_validation_completed")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cases", type=int, default=None)
    
    args = parser.parse_args()
    
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Recursively find all subdirectories containing .nii.gz files (case dirs)
    case_dirs = []
    for root, dirs, files in os.walk(args.input_dir):
        if any(f.endswith('.nii.gz') for f in files):
            case_dirs.append(root)
    case_dirs.sort()
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]
    print(f"Found {len(case_dirs)} cases")
    # Process cases
    os.makedirs(args.output_dir, exist_ok=True)
    successful = 0
    for case_dir in case_dirs:
        if process_case(case_dir, args.output_dir, args.checkpoint_dir, device):
            successful += 1
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}/{len(case_dirs)}")


if __name__ == "__main__":
    main()