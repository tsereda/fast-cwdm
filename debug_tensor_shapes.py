#!/usr/bin/env python3
"""
Debug script to identify tensor shape issues in medical image synthesis
"""

import os
import sys
import torch as th
import numpy as np
from pathlib import Path
import nibabel as nib

# Add current directory to path
sys.path.append(".")

from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def create_debug_args():
    """Create debug arguments"""
    class Args:
        pass
    
    args = Args()
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
    args.in_channels = 32
    args.out_channels = 8
    args.bottleneck_attention = False
    args.resample_2d = False
    args.additive_skips = False
    args.use_freq = False
    args.predict_xstart = True
    args.diffusion_steps = 10
    args.noise_schedule = "linear"
    args.timestep_respacing = ""
    args.use_kl = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    args.sample_schedule = "sampled"
    args.mode = 'i2i'
    args.dataset = "brats"
    
    return args

def debug_tensor_shapes():
    """Debug tensor shapes and model loading"""
    print("=== TENSOR SHAPE DEBUG SESSION ===")
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test DWT/IDWT
    print("\n1. Testing DWT/IDWT initialization...")
    try:
        dwt = DWT_3D("haar")
        idwt = IDWT_3D("haar")
        print("✅ DWT/IDWT initialized successfully")
    except Exception as e:
        print(f"❌ DWT/IDWT initialization failed: {e}")
        return
    
    # Test model creation
    print("\n2. Testing model creation...")
    try:
        args = create_debug_args()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        print(f"✅ Model created: {type(model)}")
        print(f"✅ Diffusion created: {type(diffusion)}")
        print(f"   Diffusion timesteps: {diffusion.num_timesteps}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Test model loading
    print("\n3. Testing model loading...")
    checkpoint_path = Path("./checkpoints/brats_t1n_BEST_sampled_10.pt")
    if checkpoint_path.exists():
        try:
            state_dict = dist_util.load_state_dict(str(checkpoint_path), map_location="cpu")
            print(f"✅ Checkpoint loaded: {len(state_dict)} parameters")
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("✅ Model loaded and moved to device")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Test tensor creation
    print("\n4. Testing tensor creation...")
    try:
        # Create dummy input tensors
        dummy_input = th.randn(1, 1, 224, 224, 155, device=device)
        print(f"✅ Dummy input created: {dummy_input.shape}")
        
        # Test DWT
        dwt_outputs = dwt(dummy_input)
        print(f"✅ DWT successful: {len(dwt_outputs)} components")
        print(f"   First component shape: {dwt_outputs[0].shape}")
        
        # Test conditioning tensor creation
        cond_components = []
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt_outputs
        modality_cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        cond_components.append(modality_cond)
        print(f"✅ Single modality conditioning: {modality_cond.shape}")
        
        # Simulate 3 modalities
        cond_tensor = th.cat([modality_cond, modality_cond, modality_cond], dim=1)
        print(f"✅ Full conditioning tensor: {cond_tensor.shape}")
        
        # Test noise tensor
        noise = th.randn(1, 8, 112, 112, 80, device=device)
        print(f"✅ Noise tensor created: {noise.shape}")
        
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        return
    
    # Test model forward pass
    print("\n5. Testing model forward pass...")
    try:
        # Create test timesteps
        timesteps = th.tensor([0], device=device)
        print(f"✅ Timesteps created: {timesteps.shape}, values: {timesteps}")
        
        # Test model forward
        with th.no_grad():
            output = model(noise, timesteps, cond=cond_tensor)
            print(f"✅ Model forward successful: {output.shape}")
            
    except Exception as e:
        print(f"❌ Model forward failed: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test sampling loop (just one step)
    print("\n6. Testing sampling loop...")
    try:
        diffusion.mode = 'i2i'
        
        # Test one step of sampling
        model_kwargs = {}
        
        # Create a simple wrapper for the model
        def model_fn(x, t, **kwargs):
            return model(x, t, **kwargs)
        
        with th.no_grad():
            # Test p_mean_variance
            out = diffusion.p_mean_variance(
                model_fn, noise, timesteps, cond=cond_tensor, clip_denoised=True
            )
            print(f"✅ p_mean_variance successful")
            
    except Exception as e:
        print(f"❌ Sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== DEBUG COMPLETE ===")
    print("All tests passed! The issue might be in the specific data or model configuration.")

if __name__ == "__main__":
    debug_tensor_shapes()