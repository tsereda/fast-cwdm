"""
Complete missing modalities in pseudo-validation dataset.
This is the core script that bridges synthesis and segmentation evaluation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F
import glob

sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D


def detect_missing_modality(case_dir):
    """Detect which modality is missing in a case directory."""
    modalities = ['t1n', 't1c', 't2w', 't2f']
    
    # Check for marker file first
    for modality in modalities:
        marker_file = os.path.join(case_dir, f"missing_{modality}.txt")
        if os.path.exists(marker_file):
            return modality
    
    # Fallback: check which modality file is missing
    case_name = os.path.basename(case_dir)
    missing_modalities = []
    
    for modality in modalities:
        expected_file = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        if not os.path.exists(expected_file):
            missing_modalities.append(modality)
    
    if len(missing_modalities) == 1:
        return missing_modalities[0]
    elif len(missing_modalities) == 0:
        print(f"Warning: No missing modality found in {case_dir}")
        return None
    else:
        print(f"Warning: Multiple missing modalities in {case_dir}: {missing_modalities}")
        return missing_modalities[0]  # Take the first one


def load_modalities(case_dir, case_name, missing_modality):
    """Load the available modalities for conditioning."""
    modalities = ['t1n', 't1c', 't2w', 't2f']
    available_modalities = [m for m in modalities if m != missing_modality]
    
    loaded_modalities = {}
    
    for modality in available_modalities:
        file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        if os.path.exists(file_path):
            # Load and preprocess like in BRATSVolumes
            img = nib.load(file_path).get_fdata()
            
            # Clip and normalize
            img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
            img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))
            
            # Convert to tensor and pad to expected size
            img_tensor = th.zeros(1, 240, 240, 160)
            img_tensor[:, :, :, :155] = th.tensor(img_normalized)
            img_tensor = img_tensor[:, 8:-8, 8:-8, :]  # Crop like in BRATSVolumes
            
            loaded_modalities[modality] = img_tensor.float()
        else:
            raise FileNotFoundError(f"Expected modality file not found: {file_path}")
    
    return loaded_modalities


def find_model_checkpoint(missing_modality, checkpoint_dir="/data/checkpoints"):
    """Find the latest checkpoint for the missing modality."""
    # Look for model files matching the pattern
    pattern = f"brats_{missing_modality}_*_*.pt"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found for modality {missing_modality} in {checkpoint_dir}")
    
    # Sort by iteration number (extract from filename)
    def extract_iteration(filename):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        try:
            return int(parts[2])  # brats_t1n_001000_direct_1000.pt -> 001000
        except (IndexError, ValueError):
            return 0
    
    checkpoint_files.sort(key=extract_iteration, reverse=True)
    selected_checkpoint = checkpoint_files[0]
    
    print(f"Selected checkpoint for {missing_modality}: {selected_checkpoint}")
    return selected_checkpoint


def synthesize_missing_modality(available_modalities, missing_modality, model_path, device):
    """Synthesize the missing modality using the trained model."""
    print(f"Synthesizing {missing_modality}...")
    
    # Create model and diffusion
    args = create_default_args()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    
    # Load model weights
    print(f"Loading model from: {model_path}")
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    
    # Setup wavelet transforms
    dwt = DWT_3D("haar")
    idwt = IDWT_3D("haar")
    
    # Get available modalities in consistent order
    modality_order = ['t1n', 't1c', 't2w', 't2f']
    available_order = [m for m in modality_order if m != missing_modality]
    
    # Move tensors to device
    cond_tensors = []
    for modality in available_order:
        tensor = available_modalities[modality].to(device)
        cond_tensors.append(tensor)
    
    # Create conditioning vector using DWT
    cond_list = []
    for tensor in cond_tensors:
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(tensor)
        modality_cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        cond_list.append(modality_cond)
    
    # Concatenate all conditioning modalities
    cond = th.cat(cond_list, dim=1)
    
    # Generate noise
    noise = th.randn(1, 8, 112, 112, 80).to(device)
    
    # Sample from model
    model_kwargs = {}
    with th.no_grad():
        sample = diffusion.p_sample_loop(
            model=model,
            shape=noise.shape,
            noise=noise,
            cond=cond,
            clip_denoised=True,
            model_kwargs=model_kwargs
        )
    
    # Convert back to spatial domain
    B, _, D, H, W = sample.size()
    sample = idwt(
        sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
        sample[:, 1, :, :, :].view(B, 1, D, H, W),
        sample[:, 2, :, :, :].view(B, 1, D, H, W),
        sample[:, 3, :, :, :].view(B, 1, D, H, W),
        sample[:, 4, :, :, :].view(B, 1, D, H, W),
        sample[:, 5, :, :, :].view(B, 1, D, H, W),
        sample[:, 6, :, :, :].view(B, 1, D, H, W),
        sample[:, 7, :, :, :].view(B, 1, D, H, W)
    )
    
    # Post-process
    sample[sample <= 0] = 0
    sample[sample >= 1] = 1
    
    # Use first available modality as mask
    mask_modality = cond_tensors[0]
    sample[mask_modality == 0] = 0  # Zero out non-brain parts
    
    if len(sample.shape) == 5:
        sample = sample.squeeze(dim=1)
    
    # Crop to original resolution
    sample = sample[:, :, :, :155]
    
    return sample


def create_default_args():
    """Create default arguments for model creation."""
    class Args:
        pass
    
    args = Args()
    defaults = model_and_diffusion_defaults()
    
    # Set the required parameters
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
    args.in_channels = 32  # 8 + 8*3 for conditioning
    args.out_channels = 8
    args.bottleneck_attention = False
    args.resample_2d = False
    args.additive_skips = False
    args.use_freq = False
    args.predict_xstart = True
    args.diffusion_steps = 1000
    args.noise_schedule = "linear"
    args.timestep_respacing = ""
    args.use_kl = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    args.sample_schedule = "direct"
    args.mode = 'i2i'
    args.dataset = "brats"  # Added to fix missing attribute error
    return args


def complete_case(case_dir, output_dir, device):
    """Complete a single case by synthesizing the missing modality."""
    case_name = os.path.basename(case_dir)
    
    # Detect missing modality
    missing_modality = detect_missing_modality(case_dir)
    if missing_modality is None:
        print(f"Skipping {case_name}: No missing modality detected")
        return False
    
    print(f"Processing {case_name}: missing {missing_modality}")
    
    try:
        # Load available modalities
        available_modalities = load_modalities(case_dir, case_name, missing_modality)
        
        # Find model checkpoint
        model_path = find_model_checkpoint(missing_modality)
        
        # Synthesize missing modality
        synthesized = synthesize_missing_modality(available_modalities, missing_modality, model_path, device)
        
        # Create output directory
        output_case_dir = os.path.join(output_dir, case_name)
        os.makedirs(output_case_dir, exist_ok=True)
        
        # Copy existing modalities
        for filename in os.listdir(case_dir):
            if filename.endswith('.nii.gz') and not filename.startswith('missing_'):
                src_file = os.path.join(case_dir, filename)
                dst_file = os.path.join(output_case_dir, filename)
                nib.save(nib.load(src_file), dst_file)
        
        # Save synthesized modality
        synthesized_path = os.path.join(output_case_dir, f"{case_name}-{missing_modality}.nii.gz")
        synthesized_np = synthesized.detach().cpu().numpy()[0]  # Remove batch dimension
        
        # Get reference image for proper header/affine
        reference_files = [f for f in os.listdir(case_dir) if f.endswith('.nii.gz') and 't1n' in f or 't1c' in f or 't2w' in f or 't2f' in f]
        if reference_files:
            reference_img = nib.load(os.path.join(case_dir, reference_files[0]))
            
            # Pad back to original size if needed
            original_shape = reference_img.shape
            if synthesized_np.shape != original_shape:
                # Pad the synthesized image to match original dimensions
                padded = np.zeros(original_shape)
                padded[8:232, 8:232, :155] = synthesized_np  # Reverse the cropping
                synthesized_np = padded
            
            synthesized_img = nib.Nifti1Image(synthesized_np, reference_img.affine, reference_img.header)
        else:
            synthesized_img = nib.Nifti1Image(synthesized_np, np.eye(4))
        
        nib.save(synthesized_img, synthesized_path)
        print(f"Saved synthesized {missing_modality} to {synthesized_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {case_name}: {e}")
        return False


def main():
    """Main function to complete all cases in pseudo-validation dataset."""
    parser = argparse.ArgumentParser(description="Complete missing modalities in pseudo-validation dataset")
    parser.add_argument("--input_dir", default="./datasets/BRATS2023/pseudo_validation", 
                       help="Directory containing pseudo-validation data with missing modalities")
    parser.add_argument("--output_dir", default="./datasets/BRATS2023/pseudo_validation_completed",
                       help="Output directory for completed dataset")
    parser.add_argument("--checkpoint_dir", default="/data/checkpoints",
                       help="Directory containing trained model checkpoints")
    parser.add_argument("--device", default="cuda:0", help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all cases
    case_dirs = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d))]
    case_dirs.sort()
    
    print(f"Found {len(case_dirs)} cases to process")
    
    successful = 0
    failed = 0
    
    for case_dir_name in case_dirs:
        case_dir = os.path.join(args.input_dir, case_dir_name)
        success = complete_case(case_dir, args.output_dir, device)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted processing:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(case_dirs)}")
    
    if successful > 0:
        print(f"\nCompleted dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()