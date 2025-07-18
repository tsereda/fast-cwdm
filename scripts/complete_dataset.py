#!/usr/bin/env python3
"""
Enhanced Medical Image Synthesis for Missing Modality Completion

This script synthesizes missing MRI modalities using trained diffusion models.
It processes BraTS dataset cases and completes missing modalities (t1n, t1c, t2w, t2f).
"""

import argparse
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import Tensor

# Add current directory to path for imports
sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D


# Constants
MODALITIES = ['t1n', 't1c', 't2w', 't2f']
DEFAULT_DIFFUSION_STEPS = 1000
DEFAULT_SAMPLE_SCHEDULE = 'direct'
IMAGE_CROP_SIZE = (224, 224, 155)
PADDED_SIZE = (240, 240, 160)
CROP_MARGIN = 8

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthesis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for model parameters"""
    
    def __init__(self):
        self.image_size = 224
        self.num_channels = 64
        self.num_res_blocks = 2
        self.channel_mult = "1,2,2,4,4"
        self.learn_sigma = False
        self.class_cond = False
        self.use_checkpoint = False
        self.attention_resolutions = ""
        self.num_heads = 1
        self.num_head_channels = -1
        self.num_heads_upsample = -1
        self.use_scale_shift_norm = False
        self.dropout = 0.0
        self.resblock_updown = True
        self.use_fp16 = False
        self.use_new_attention_order = False
        self.dims = 3
        self.num_groups = 32
        self.in_channels = 32  # 8 + 8*3 for conditioning
        self.out_channels = 8
        self.bottleneck_attention = False
        self.resample_2d = False
        self.additive_skips = False
        self.use_freq = False
        self.predict_xstart = True
        self.noise_schedule = "linear"
        self.timestep_respacing = ""
        self.use_kl = False
        self.rescale_timesteps = False
        self.rescale_learned_sigmas = False
        self.mode = 'i2i'
        self.dataset = "brats"


class ModalityDetector:
    """Handles detection of missing modalities in case directories"""
    
    @staticmethod
    def get_case_name_from_files(case_dir: Path) -> Optional[str]:
        """Extract actual case name from existing files"""
        for file in case_dir.glob("*.nii.gz"):
            if file.name.startswith('BraTS-'):
                # Extract case name from filename
                parts = file.stem.split('-')
                if len(parts) >= 4:
                    return "-".join(parts[:-1])
        return None
    
    @staticmethod
    def detect_missing_modality(case_dir: Path) -> Optional[str]:
        """Detect which modality is missing in a case directory"""
        case_dir = Path(case_dir)
        
        # Check for marker files first
        for modality in MODALITIES:
            marker_file = case_dir / f"missing_{modality}.txt"
            if marker_file.exists():
                return modality
        
        # Fallback: check which modality file is missing
        actual_case_name = ModalityDetector.get_case_name_from_files(case_dir)
        if not actual_case_name:
            logger.warning(f"Could not determine actual case name in {case_dir}")
            return None
        
        missing_modalities = []
        for modality in MODALITIES:
            expected_file = case_dir / f"{actual_case_name}-{modality}.nii.gz"
            if not expected_file.exists():
                missing_modalities.append(modality)
        
        if len(missing_modalities) == 1:
            return missing_modalities[0]
        elif len(missing_modalities) == 0:
            logger.warning(f"No missing modality found in {case_dir}")
            return None
        else:
            logger.warning(f"Multiple missing modalities in {case_dir}: {missing_modalities}")
            return missing_modalities[0]  # Return first one


class ImagePreprocessor:
    """Handles image loading and preprocessing"""
    
    @staticmethod
    def load_and_preprocess_image(file_path: Path) -> Tensor:
        """Load and preprocess a single image"""
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Load image
        img = nib.load(str(file_path)).get_fdata()
        
        # Clip and normalize
        img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
        img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))
        
        # Convert to tensor and pad
        img_tensor = th.zeros(1, *PADDED_SIZE)
        img_tensor[:, :, :, :IMAGE_CROP_SIZE[2]] = th.tensor(img_normalized)
        
        # Crop like in BRATSVolumes
        img_tensor = img_tensor[:, CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN, :]
        
        return img_tensor.float()
    
    @staticmethod
    def load_modalities(case_dir: Path, case_name: str, missing_modality: str) -> Dict[str, Tensor]:
        """Load all available modalities for conditioning"""
        available_modalities = [m for m in MODALITIES if m != missing_modality]
        loaded_modalities = {}
        
        for modality in available_modalities:
            file_path = case_dir / f"{case_name}-{modality}.nii.gz"
            loaded_modalities[modality] = ImagePreprocessor.load_and_preprocess_image(file_path)
        
        return loaded_modalities


class CheckpointManager:
    """Manages model checkpoint loading and parameter parsing"""
    
    @staticmethod
    def parse_checkpoint_parameters(model_path: Path) -> Dict[str, Union[str, int]]:
        """Parse checkpoint filename to extract parameters"""
        basename = model_path.name
        parts = basename.split('_')
        
        # Default values
        sample_schedule = DEFAULT_SAMPLE_SCHEDULE
        diffusion_steps = DEFAULT_DIFFUSION_STEPS
        
        try:
            # Handle BEST pattern: brats_t1n_BEST_direct_1000.pt
            if "_BEST_" in basename:
                if len(parts) >= 4:
                    sample_schedule = parts[3]
                if len(parts) >= 5:
                    steps_str = parts[4].split('.')[0]
                    diffusion_steps = int(steps_str)
            # Handle regular pattern: brats_t1n_010000_direct_1000.pt
            elif len(parts) >= 5:
                sample_schedule = parts[3]
                steps_str = parts[4].split('.')[0]
                diffusion_steps = int(steps_str)
            elif len(parts) >= 4:
                sample_schedule = parts[3].split('.')[0]
                if sample_schedule.isdigit():
                    diffusion_steps = int(sample_schedule)
                    sample_schedule = DEFAULT_SAMPLE_SCHEDULE
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing checkpoint parameters: {e}")
        
        logger.info(f"Parsed {basename} → schedule={sample_schedule}, steps={diffusion_steps}")
        return {
            'sample_schedule': sample_schedule,
            'diffusion_steps': diffusion_steps
        }
    
    @staticmethod
    def find_model_checkpoint(missing_modality: str, checkpoint_dir: Path) -> Path:
        """Find the latest checkpoint for the missing modality"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Try different naming patterns
        patterns = [
            f"brats_{missing_modality}_BEST_*.pt",
            f"brats_{missing_modality}_*_*.pt",
            f"*{missing_modality}*.pt"
        ]
        
        checkpoint_files = []
        for pattern in patterns:
            files = list(checkpoint_dir.glob(pattern))
            if files:
                checkpoint_files.extend(files)
                break
        
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint found for modality {missing_modality} in {checkpoint_dir}"
            )
        
        # Sort by priority and iteration number
        def extract_priority(filename: Path) -> Tuple[int, str, int]:
            basename = filename.name
            if "_BEST_" in basename:
                return 999999, DEFAULT_SAMPLE_SCHEDULE, DEFAULT_DIFFUSION_STEPS
            
            parts = basename.split('_')
            try:
                iter_num = int(parts[2]) if len(parts) > 2 else 0
                schedule = parts[3] if len(parts) > 3 else DEFAULT_SAMPLE_SCHEDULE
                steps_str = parts[4].split('.')[0] if len(parts) > 4 else str(DEFAULT_DIFFUSION_STEPS)
                steps = int(steps_str)
                return iter_num, schedule, steps
            except (ValueError, IndexError):
                return 0, DEFAULT_SAMPLE_SCHEDULE, DEFAULT_DIFFUSION_STEPS
        
        # Sort by iteration number (BEST models get highest priority)
        checkpoint_files.sort(key=extract_priority, reverse=True)
        selected_checkpoint = checkpoint_files[0]
        
        logger.info(f"Selected checkpoint for {missing_modality}: {selected_checkpoint}")
        return selected_checkpoint
    
    @staticmethod
    def create_args_from_checkpoint(model_path: Path) -> argparse.Namespace:
        """Create arguments object from checkpoint parameters"""
        params = CheckpointManager.parse_checkpoint_parameters(model_path)
        config = Config()
        
        # Create namespace object
        args = argparse.Namespace()
        
        # Set all config attributes
        for attr, value in vars(config).items():
            setattr(args, attr, value)
        
        # Set checkpoint-specific parameters
        args.sample_schedule = params['sample_schedule']
        args.diffusion_steps = params['diffusion_steps']
        
        return args


class DiffusionSynthesizer:
    """Handles diffusion model synthesis"""
    
    def __init__(self, device: th.device):
        self.device = device
        self.dwt = DWT_3D("haar")
        self.idwt = IDWT_3D("haar")
    
    def synthesize_missing_modality(
        self,
        available_modalities: Dict[str, Tensor],
        missing_modality: str,
        model_path: Path,
    ) -> Tensor:
        """Synthesize the missing modality using the trained model"""
        logger.info(f"Synthesizing {missing_modality}...")
        
        # Create model and diffusion
        args = CheckpointManager.create_args_from_checkpoint(model_path)
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        diffusion.mode = 'i2i'
        
        # Load model weights
        logger.info(f"Loading model from: {model_path}")
        state_dict = dist_util.load_state_dict(str(model_path), map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Prepare conditioning
        cond = self._prepare_conditioning(available_modalities, missing_modality)
        
        # Generate and sample
        noise = th.randn(1, 8, 112, 112, 80, device=self.device)
        
        with th.no_grad():
            logger.info(f"Running {diffusion.num_timesteps}-step sampling...")
            sample = diffusion.p_sample_loop(
                model=model,
                shape=noise.shape,
                noise=noise,
                cond=cond,
                clip_denoised=True,
                model_kwargs={}
            )
        
        # Convert back to spatial domain
        sample = self._convert_to_spatial_domain(sample)
        
        # Post-process
        sample = self._post_process_sample(sample, available_modalities)
        
        return sample
    
    def _prepare_conditioning(self, available_modalities: Dict[str, Tensor], missing_modality: str) -> Tensor:
        """Prepare conditioning vector from available modalities"""
        # Get available modalities in consistent order
        available_order = [m for m in MODALITIES if m != missing_modality]
        
        # Move tensors to device and ensure 5D shape
        cond_tensors = []
        for modality in available_order:
            tensor = available_modalities[modality].to(self.device)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(1)  # Add channel dimension
            cond_tensors.append(tensor)
        
        # Create conditioning vector using DWT
        cond_list = []
        for tensor in cond_tensors:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(tensor)
            modality_cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
            cond_list.append(modality_cond)
        
        return th.cat(cond_list, dim=1)
    
    def _convert_to_spatial_domain(self, sample: Tensor) -> Tensor:
        """Convert sample from wavelet domain to spatial domain"""
        B, _, D, H, W = sample.size()
        sample = self.idwt(
            sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
            sample[:, 1, :, :, :].view(B, 1, D, H, W),
            sample[:, 2, :, :, :].view(B, 1, D, H, W),
            sample[:, 3, :, :, :].view(B, 1, D, H, W),
            sample[:, 4, :, :, :].view(B, 1, D, H, W),
            sample[:, 5, :, :, :].view(B, 1, D, H, W),
            sample[:, 6, :, :, :].view(B, 1, D, H, W),
            sample[:, 7, :, :, :].view(B, 1, D, H, W)
        )
        return sample
    
    def _post_process_sample(self, sample: Tensor, available_modalities: Dict[str, Tensor]) -> Tensor:
        """Post-process the generated sample"""
        # Clamp values
        sample = th.clamp(sample, 0, 1)
        
        # Apply mask from first available modality
        if available_modalities:
            first_modality = list(available_modalities.values())[0].to(self.device)
            if first_modality.dim() == 4:
                first_modality = first_modality.unsqueeze(1)
            sample[first_modality == 0] = 0
        
        # Remove extra dimensions and crop
        if sample.dim() == 5:
            sample = sample.squeeze(dim=1)
        
        # Crop to original resolution
        sample = sample[:, :, :, :IMAGE_CROP_SIZE[2]]
        
        return sample


class CaseProcessor:
    """Handles processing of individual cases"""
    
    def __init__(self, synthesizer: DiffusionSynthesizer, checkpoint_dir: Path):
        self.synthesizer = synthesizer
        self.checkpoint_dir = checkpoint_dir
    
    def process_case(self, case_dir: Path, output_dir: Path) -> bool:
        """Process a single case by synthesizing the missing modality"""
        try:
            # Get case name and missing modality
            case_name = ModalityDetector.get_case_name_from_files(case_dir)
            if not case_name:
                logger.warning(f"Skipping {case_dir.name}: Could not determine case name")
                return False
            
            missing_modality = ModalityDetector.detect_missing_modality(case_dir)
            if not missing_modality:
                logger.warning(f"Skipping {case_name}: No missing modality detected")
                return False
            
            logger.info(f"Processing {case_name}: missing {missing_modality}")
            
            # Load available modalities
            available_modalities = ImagePreprocessor.load_modalities(
                case_dir, case_name, missing_modality
            )
            
            # Find and load model
            model_path = CheckpointManager.find_model_checkpoint(
                missing_modality, self.checkpoint_dir
            )
            
            # Synthesize missing modality
            synthesized = self.synthesizer.synthesize_missing_modality(
                available_modalities, missing_modality, model_path
            )
            
            # Save results
            self._save_results(case_dir, output_dir, case_name, missing_modality, synthesized)
            
            logger.info(f"Successfully processed {case_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {case_dir.name}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _save_results(
        self,
        case_dir: Path,
        output_dir: Path,
        case_name: str,
        missing_modality: str,
        synthesized: Tensor
    ):
        """Save processing results"""
        # Create output directory
        output_case_dir = output_dir / case_name
        output_case_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy existing modalities
        for file in case_dir.glob("*.nii.gz"):
            if not file.name.startswith('missing_'):
                dst_file = output_case_dir / file.name
                nib.save(nib.load(str(file)), str(dst_file))
        
        # Save synthesized modality
        synthesized_path = output_case_dir / f"{case_name}-{missing_modality}.nii.gz"
        synthesized_np = synthesized.detach().cpu().numpy()[0]
        
        # Get reference for proper header/affine
        reference_files = [f for f in case_dir.glob("*.nii.gz") 
                          if any(mod in f.name for mod in MODALITIES)]
        
        if reference_files:
            reference_img = nib.load(str(reference_files[0]))
            original_shape = reference_img.shape
            
            # Pad back to original size if needed
            if synthesized_np.shape != original_shape:
                padded = np.zeros(original_shape)
                padded[
                    CROP_MARGIN:CROP_MARGIN + synthesized_np.shape[0],
                    CROP_MARGIN:CROP_MARGIN + synthesized_np.shape[1],
                    :synthesized_np.shape[2]
                ] = synthesized_np
                synthesized_np = padded
            
            synthesized_img = nib.Nifti1Image(
                synthesized_np, reference_img.affine, reference_img.header
            )
        else:
            synthesized_img = nib.Nifti1Image(synthesized_np, np.eye(4))
        
        nib.save(synthesized_img, str(synthesized_path))
        logger.info(f"Saved synthesized {missing_modality} to {synthesized_path}")


def setup_environment(args: argparse.Namespace) -> Tuple[th.device, Path, Path]:
    """Setup environment and validate paths"""
    # Set random seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup device
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return device, input_dir, output_dir


def find_case_directories(input_dir: Path) -> List[Path]:
    """Find all case directories containing BraTS data"""
    case_dirs = []
    
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        # Look for directories containing BraTS NIfTI files
        if any(f.endswith('.nii.gz') and f.startswith('BraTS-GLI-') for f in files):
            case_dirs.append(root_path)
    
    return sorted(list(set(case_dirs)))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Complete missing modalities in pseudo-validation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str,
        default="./datasets/BRATS2023/pseudo_validation",
        help="Directory containing pseudo-validation data with missing modalities"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./datasets/BRATS2023/pseudo_validation_completed",
        help="Output directory for completed dataset"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str,
        default="/data/checkpoints",
        help="Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="cuda:0", 
        help="Device to run inference on"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--max_cases", 
        type=int, 
        default=None,
        help="Maximum number of cases to process (for debugging)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser


def main():
    """Main function to complete all cases in pseudo-validation dataset"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Setup environment
        device, input_dir, output_dir = setup_environment(args)
        
        # Find case directories
        case_dirs = find_case_directories(input_dir)
        if args.max_cases is not None:
            case_dirs = case_dirs[:args.max_cases]
        
        logger.info(f"Found {len(case_dirs)} cases to process")
        
        # Initialize synthesizer and processor
        synthesizer = DiffusionSynthesizer(device)
        processor = CaseProcessor(synthesizer, Path(args.checkpoint_dir))
        
        # Process all cases
        successful = 0
        failed = 0
        
        for case_dir in case_dirs:
            success = processor.process_case(case_dir, output_dir)
            if success:
                successful += 1
            else:
                failed += 1
        
        # Print summary
        logger.info(f"\nProcessing Summary:")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {len(case_dirs)}")
        
        if successful > 0:
            logger.info(f"Completed dataset saved to: {output_dir}")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()