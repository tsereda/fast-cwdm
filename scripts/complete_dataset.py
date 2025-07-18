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
    # Get the actual case name from an existing file to avoid issues with parent directory names
    actual_case_name = None
    for f in os.listdir(case_dir):
        if f.endswith('.nii.gz'):
            # Example: BraTS-GLI-00721-000-t1c.nii.gz -> BraTS-GLI-00721-000
            actual_case_name = "-".join(f.split('-')[:-1]) 
            break
    
    if actual_case_name is None:
        print(f"Warning: Could not determine actual case name in {case_dir}")
        return None

    missing_modalities = []
    
    for modality in modalities:
        expected_file = os.path.join(case_dir, f"{actual_case_name}-{modality}.nii.gz")
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


def load_modalities(case_dir, actual_case_name, missing_modality): # Changed case_name to actual_case_name
    """Load the available modalities for conditioning."""
    modalities = ['t1n', 't1c', 't2w', 't2f']
    available_modalities = [m for m in modalities if m != missing_modality]
    
    loaded_modalities = {}
    
    for modality in available_modalities:
        file_path = os.path.join(case_dir, f"{actual_case_name}-{modality}.nii.gz") # Use actual_case_name
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


def find_model_checkpoint(missing_modality, checkpoint_dir="./checkpoints"):
    """Find the latest checkpoint for the missing modality, handling BEST naming pattern."""
    import glob
    import os
    # Try different naming patterns
    patterns = [
        f"brats_{missing_modality}_BEST_*.pt",      # BEST pattern
        f"brats_{missing_modality}_*_*.pt",         # Regular pattern
        f"*{missing_modality}*.pt"                  # Fallback pattern
    ]
    checkpoint_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if files:
            checkpoint_files.extend(files)
            break  # Use first pattern that finds files
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found for modality {missing_modality} in {checkpoint_dir}")
    # Helper to extract iteration number and schedule info
    def extract_info(filename):
        basename = os.path.basename(filename)
        # Handle BEST pattern: brats_t1n_BEST_direct_1000.pt
        if "_BEST_" in basename:
            parts = basename.split('_')
            try:
                schedule = parts[3] if len(parts) > 3 else 'direct'
                steps_str = parts[4].split('.')[0] if len(parts) > 4 else '1000'
                steps = int(steps_str)
                return 999999, schedule, steps  # High priority for BEST models
            except (ValueError, IndexError):
                return 999999, 'direct', 1000
        # Handle regular pattern: brats_t1n_010000_direct_1000.pt  
        parts = basename.split('_')
        try:
            iter_num = int(parts[2]) if len(parts) > 2 else 0
            schedule = parts[3] if len(parts) > 3 else 'direct'
            steps_str = parts[4].split('.')[0] if len(parts) > 4 else '1000'
            steps = int(steps_str)
            return iter_num, schedule, steps
        except (ValueError, IndexError):
            return 0, 'direct', 1000
    # Sort by iteration number (BEST models get highest priority)
    checkpoint_info = [(extract_info(f), f) for f in checkpoint_files]
    checkpoint_info.sort(reverse=True, key=lambda x: x[0][0])  # Sort by iteration number
    selected_checkpoint = checkpoint_info[0][1]
    iter_num, schedule, steps = checkpoint_info[0][0]
    print(f"Selected checkpoint for {missing_modality}: {selected_checkpoint}")
    print(f"    Schedule: {schedule}, Steps: {steps}, Iteration: {iter_num}")
    return selected_checkpoint


def synthesize_missing_modality(available_modalities, missing_modality, model_path, device):
    """Synthesize the missing modality using the trained model with dynamic parameters."""
    print(f"Synthesizing {missing_modality}...")

    # 🔥 Parse checkpoint parameters automatically
    checkpoint_params = parse_checkpoint_parameters(model_path)
    print(f"[INFERENCE] Using {checkpoint_params['sample_schedule']} schedule with {checkpoint_params['diffusion_steps']} steps")

    # Create model and diffusion with detected parameters
    args = create_args_from_checkpoint(model_path)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'

    # Load model weights on CPU first, then move to CUDA (prevents device-side assert)
    print(f"Loading model from: {model_path}")
    state_dict = dist_util.load_state_dict(model_path, map_location="cpu")
    print("[DEBUG] Model state_dict keys:", list(state_dict.keys())[:10], "... total:", len(state_dict))
    model.load_state_dict(state_dict)
    print(f"[DEBUG] Moving model to device: {device}")
    model.to(device)
    model.eval()

    # Setup wavelet transforms
    dwt = DWT_3D("haar")
    idwt = IDWT_3D("haar")

    # Get available modalities in consistent order
    modality_order = ['t1n', 't1c', 't2w', 't2f']
    available_order = [m for m in modality_order if m != missing_modality]

    # Move tensors to device and ensure 5D shape [B, C, D, H, W]
    cond_tensors = []
    for modality in available_order:
        tensor = available_modalities[modality]
        print(f"[DEBUG] Input tensor for {modality}: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min().item()}, max={tensor.max().item()}")
        tensor = tensor.to(device)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)  # Add channel dimension: [B, 1, D, H, W]
        cond_tensors.append(tensor)
    print(f"[DEBUG] cond_tensors shapes: {[t.shape for t in cond_tensors]}")

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

    # Sample from model using detected parameters
    model_kwargs = {}
    with th.no_grad():
        print(f"[SAMPLING] Running {diffusion.num_timesteps}-step sampling...")
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


def parse_checkpoint_parameters(model_path):
    """
    Parse checkpoint filename to extract sample_schedule and diffusion_steps.
    Updated to handle BEST pattern: brats_t1n_BEST_direct_1000.pt
    """
    import os
    basename = os.path.basename(model_path)
    parts = basename.split('_')
    # Default values
    sample_schedule = 'direct'
    diffusion_steps = 1000
    # Handle BEST pattern: brats_t1n_BEST_direct_1000.pt
    if "_BEST_" in basename:
        try:
            if len(parts) >= 4:
                sample_schedule = parts[3]  # 'direct'
            if len(parts) >= 5:
                steps_str = parts[4].split('.')[0]  # '1000'
                diffusion_steps = int(steps_str)
        except (ValueError, IndexError):
            pass
    # Handle regular pattern: brats_t1n_010000_direct_1000.pt
    elif len(parts) >= 5:
        try:
            sample_schedule = parts[3]
            steps_str = parts[4].split('.')[0]
            diffusion_steps = int(steps_str)
        except (ValueError, IndexError):
            pass
    elif len(parts) >= 4:
        try:
            sample_schedule = parts[3].split('.')[0]
            if sample_schedule.isdigit():
                diffusion_steps = int(sample_schedule)
                sample_schedule = 'direct'
        except (ValueError, IndexError):
            pass
    print(f"[CHECKPOINT] Parsed {basename} → schedule={sample_schedule}, steps={diffusion_steps}")
    return {'sample_schedule': sample_schedule, 'diffusion_steps': diffusion_steps}

def create_args_from_checkpoint(model_path):
    """
    Create an Args object for model creation, using parameters parsed from checkpoint filename.
    """
    params = parse_checkpoint_parameters(model_path)
    class Args:
        pass
    args = Args()
    defaults = model_and_diffusion_defaults()
    # Set defaults
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
    args.noise_schedule = "linear"
    args.timestep_respacing = ""
    args.use_kl = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    args.mode = 'i2i'
    args.dataset = "brats"
    # Set from checkpoint
    args.sample_schedule = params['sample_schedule']
    args.diffusion_steps = params['diffusion_steps']
    return args

def complete_case(case_dir, output_dir, device):
    """Complete a single case by synthesizing the missing modality."""
    # First, try to determine the actual case name (e.g., BraTS-GLI-00721-000)
    # by looking at the files present, not the parent directory name.
    actual_case_name = None
    for f in os.listdir(case_dir):
        if f.endswith('.nii.gz'):
            actual_case_name = "-".join(f.split('-')[:-1])
            break
            
    if actual_case_name is None:
        print(f"Skipping {os.path.basename(case_dir)}: Could not determine actual case name from files.")
        return False

    # Detect missing modality
    missing_modality = detect_missing_modality(case_dir)
    if missing_modality is None:
        print(f"Skipping {actual_case_name}: No missing modality detected")
        return False
    
    print(f"Processing {actual_case_name}: missing {missing_modality}")
    
    try:
        # Load available modalities
        available_modalities = load_modalities(case_dir, actual_case_name, missing_modality) # Pass actual_case_name
        # Find model checkpoint
        model_path = find_model_checkpoint(missing_modality)
        # Synthesize missing modality
        synthesized = synthesize_missing_modality(available_modalities, missing_modality, model_path, device)
        # Create output directory for the actual case name
        output_case_dir = os.path.join(output_dir, actual_case_name)
        os.makedirs(output_case_dir, exist_ok=True)
        # Copy existing modalities
        for filename in os.listdir(case_dir):
            if filename.endswith('.nii.gz') and not filename.startswith('missing_'):
                src_file = os.path.join(case_dir, filename)
                dst_file = os.path.join(output_case_dir, filename)
                nib.save(nib.load(src_file), dst_file)
        # Save synthesized modality
        synthesized_path = os.path.join(output_case_dir, f"{actual_case_name}-{missing_modality}.nii.gz")
        synthesized_np = synthesized.detach().cpu().numpy()[0]  # Remove batch dimension
        # Get reference image for proper header/affine
        reference_files = [f for f in os.listdir(case_dir) if f.endswith('.nii.gz') and 
                           any(mod in f for mod in ['t1n', 't1c', 't2w', 't2f'])]
        if reference_files:
            reference_img = nib.load(os.path.join(case_dir, reference_files[0]))
            # Pad back to original size if needed
            original_shape = reference_img.shape
            if synthesized_np.shape != original_shape:
                # Pad the synthesized image to match original dimensions
                padded = np.zeros(original_shape)
                # Assuming 8-pixel crop on each side for x and y, and 5-slice padding for z (160 -> 155)
                # The crop in BRATSVolumes is 8:-8 for x and y. So, add 8 back to each side.
                # The z-axis is cropped from 160 to 155, so we need to put it back into the first 155 slices.
                padded[8:8+synthesized_np.shape[0], 8:8+synthesized_np.shape[1], :synthesized_np.shape[2]] = synthesized_np
                synthesized_np = padded
            synthesized_img = nib.Nifti1Image(synthesized_np, reference_img.affine, reference_img.header)
        else:
            synthesized_img = nib.Nifti1Image(synthesized_np, np.eye(4))
        nib.save(synthesized_img, synthesized_path)
        print(f"Saved synthesized {missing_modality} to {synthesized_path}")
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {actual_case_name}: {e}")
        if 'CUDA error: device-side assert triggered' in str(e):
            print('CUDA device-side assert triggered. For debugging, try running with CUDA_LAUNCH_BLOCKING=1')
        traceback.print_exc()
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
    parser.add_argument("--max_cases", type=int, default=None, help="Maximum number of cases to process (for debugging)")

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
    # The `input_dir` itself might be the "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData" directory
    # or it might contain multiple such directories.
    # We need to iterate through the actual patient directories within `input_dir`.
    
    # Let's assume input_dir points to the parent of patient directories.
    # Example: input_dir = ./datasets/BRATS2023/pseudo_validation
    # and patient data is in ./datasets/BRATS2023/pseudo_validation/BraTS-GLI-XXXXX-YYY
    
    # If `input_dir` directly contains patient folders like `BraTS-GLI-00721-000`, 
    # then `case_dirs` should be `[d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]`
    # as currently written in the script.
    
    # However, your example output shows:
    # `Warning: Multiple missing modalities in ./datasets/BRATS2023/pseudo_validation/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData: ['t1n', 't1c', 't2w', 't2f']`
    # This suggests that `input_dir` might be `pseudo_validation` and `ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData` is treated as a case.
    # We need to ensure that `case_dir` inside the loop refers to the *individual patient folder*.

    # Revised logic to find actual patient case directories
    all_potential_case_dirs = []
    for root, dirs, files in os.walk(args.input_dir):
        # Look for directories that contain NIfTI files that follow the BraTS naming convention
        if any(f.endswith('.nii.gz') and f.startswith('BraTS-GLI-') for f in files):
            all_potential_case_dirs.append(root)

    case_dirs = sorted(list(set(all_potential_case_dirs)))  # Ensure unique and sorted
    if args.max_cases is not None:
        case_dirs = case_dirs[:args.max_cases]

    print(f"Found {len(case_dirs)} cases to process")

    successful = 0
    failed = 0

    for case_dir in case_dirs:
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