"""
Complete evaluation pipeline for synthesis models.
Runs nnUNet segmentation and calculates Dice scores.
"""

import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient between two binary masks."""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    """Calculate average Dice coefficient across multiple labels."""
    dice = 0
    for index in range(numLabels):
        dice += dice_coefficient(y_true == index, y_pred == index)
    return dice / numLabels

def calculate_dice_scores(results_folder, ground_truth_folder):
    """Calculate Dice scores between predicted and ground truth segmentations."""
    dice_scores = []
    case_results = {}

    for file_name in os.listdir(results_folder):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            result_path = os.path.join(results_folder, file_name)
            gt_path = os.path.join(ground_truth_folder, file_name)

            if os.path.exists(gt_path):
                try:
                    result_img = nib.load(result_path)
                    gt_img = nib.load(gt_path)

                    result_data = result_img.get_fdata()
                    gt_data = gt_img.get_fdata()

                    dice = dice_coef_multilabel(gt_data, result_data, 4)
                    dice_scores.append(dice)
                    case_results[file_name] = dice
                    
                    print(f"{file_name}: Dice = {dice:.4f}")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"Ground truth for {file_name} not found!")

    if dice_scores:
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"Average Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"Number of cases: {len(dice_scores)}")
        print(f"Min Dice: {np.min(dice_scores):.4f}")
        print(f"Max Dice: {np.max(dice_scores):.4f}")
        
        return avg_dice, std_dice, case_results
    else:
        print("No valid segmentation files found for comparison.")
        return None, None, {}

def setup_nnunet_environment():
    """Setup nnUNet environment variables."""
    os.environ["nnUNet_raw"] = "/app/nnunet/raw"
    os.environ["nnUNet_preprocessed"] = "/app/nnunet/preprocessed" 
    os.environ["nnUNet_results"] = "/app/nnunet/results"
    
    # Create directories
    for path in ["/app/nnunet/raw", "/app/nnunet/preprocessed", "/app/nnunet/results"]:
        os.makedirs(path, exist_ok=True)
    
    print("nnUNet environment variables set:")
    print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")

def download_nnunet_weights():
    """Download pre-trained nnUNet weights if not available."""
    weights_dir = "/app/nnunet/results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_5"
    os.makedirs(weights_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(weights_dir, "checkpoint_final.pth")
    
    if not os.path.exists(checkpoint_path):
        print("Downloading nnUNet weights...")
        try:
            # Download using gdown (Google Drive)
            import gdown
            
            # Download checkpoint
            gdown.download("1n9dqT114udr9Qq8iYEKsJK347iHg9N88", "checkpoint_best.pth", quiet=False)
            os.rename("checkpoint_best.pth", checkpoint_path)
            
            # Download dataset.json  
            dataset_json_path = "/app/nnunet/results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
            gdown.download("1A_suxQwElucF3w1HEYg3wMo6dG9OxBHo", dataset_json_path, quiet=False)
            
            # Download plans.json
            plans_json_path = "/app/nnunet/results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"  
            gdown.download("1U2b0BTNi8zrJACReoi_W08Fe-wM394wI", plans_json_path, quiet=False)
            
            print("✅ nnUNet weights downloaded successfully")
            
        except ImportError:
            print("❌ gdown not installed. Install with: pip install gdown")
            return False
        except Exception as e:
            print(f"❌ Error downloading weights: {e}")
            return False
    else:
        print("✅ nnUNet weights already available")
    
    return True

def run_nnunet_prediction(dataset_dir, output_dir):
    """Run nnUNet prediction on the dataset."""
    
    print(f"Running nnUNet prediction...")
    print(f"Input: {dataset_dir}/imagesTr")
    print(f"Output: {output_dir}")
    
    # Create symlink in nnUNet_raw
    raw_dataset_path = "/app/nnunet/raw/Dataset137_BraTS2021"
    if os.path.exists(raw_dataset_path):
        os.remove(raw_dataset_path)
    os.symlink(os.path.abspath(dataset_dir), raw_dataset_path)
    
    # Run nnUNet prediction
    cmd = [
        "nnUNetv2_predict",
        "-i", f"{dataset_dir}/imagesTr",
        "-o", output_dir,
        "-d", "137",
        "-c", "3d_fullres", 
        "-f", "5"
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ nnUNet prediction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ nnUNet prediction failed:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesis models using segmentation performance")
    parser.add_argument("--dataset_dir", default="./Dataset137_BraTS21_Completed",
                       help="nnUNet format dataset directory")
    parser.add_argument("--output_dir", default="./segmentation_outputs",
                       help="Output directory for segmentation results")
    parser.add_argument("--skip_segmentation", action="store_true",
                       help="Skip segmentation and only calculate Dice scores")
    
    args = parser.parse_args()
    
    # Setup nnUNet environment
    setup_nnunet_environment()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        print("Run prepare_nnunet_dataset.py first!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_segmentation:
        # Download nnUNet weights if needed
        if not download_nnunet_weights():
            print("❌ Failed to setup nnUNet weights")
            return
        
        # Run nnUNet prediction
        if not run_nnunet_prediction(args.dataset_dir, args.output_dir):
            print("❌ nnUNet prediction failed")
            return
    
    # Calculate Dice scores
    ground_truth_dir = f"{args.dataset_dir}/labelsTr"
    
    if not os.path.exists(ground_truth_dir):
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        return
    
    if not os.listdir(args.output_dir):
        print(f"❌ No segmentation results found in {args.output_dir}")
        return
    
    print(f"\n🔍 Calculating Dice scores...")
    avg_dice, std_dice, case_results = calculate_dice_scores(args.output_dir, ground_truth_dir)
    
    # Save results
    if avg_dice is not None:
        results_file = "synthesis_evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write("Synthesis Model Evaluation Results\n")
            f.write("=====================================\n\n")
            f.write(f"Average Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Number of cases: {len(case_results)}\n")
            f.write(f"Min Dice: {min(case_results.values()):.4f}\n")
            f.write(f"Max Dice: {max(case_results.values()):.4f}\n\n")
            f.write("Per-case results:\n")
            for case, dice in sorted(case_results.items()):
                f.write(f"  {case}: {dice:.4f}\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Summary
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Your CWDM synthesis models achieved:")
        print(f"Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"This measures how well synthesized modalities preserve")
        print(f"segmentation-relevant information!")

if __name__ == "__main__":
    main()