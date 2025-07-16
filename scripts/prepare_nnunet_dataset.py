"""
Convert completed pseudo-validation dataset to nnUNet format for segmentation evaluation.
Adapted from Dataset137_BraTS21.py in the BraSyn tutorial.
"""

import multiprocessing
import shutil
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import os

def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    """
    Convert BraTS segmentation labels to nnUNet format.
    BraTS: 0, 1, 2, 4 -> nnUNet: 0, 1, 2, 3
    """
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            print(f"Warning: unexpected label {u} in {in_file}")

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3  # enhancing tumor
    seg_new[img_npy == 2] = 1  # edema 
    seg_new[img_npy == 1] = 2  # non-enhancing tumor core
    
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def setup_nnunet_dataset(completed_data_dir, output_base_dir="./Dataset137_BraTS21_Completed"):
    """
    Convert completed pseudo-validation dataset to nnUNet format.
    """
    
    print(f"Converting {completed_data_dir} to nnUNet format...")
    print(f"Output directory: {output_base_dir}")
    
    # Create nnUNet directory structure
    imagestr = join(output_base_dir, "imagesTr")
    labelstr = join(output_base_dir, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # Get all case directories
    case_dirs = [d for d in os.listdir(completed_data_dir) 
                if os.path.isdir(os.path.join(completed_data_dir, d))]
    case_dirs.sort()
    
    print(f"Found {len(case_dirs)} cases to convert")
    
    converted_cases = 0
    
    for case_name in case_dirs:
        case_dir = os.path.join(completed_data_dir, case_name)
        
        # Check if all required files exist
        required_files = [
            f"{case_name}-t1n.nii.gz",
            f"{case_name}-t1c.nii.gz", 
            f"{case_name}-t2w.nii.gz",
            f"{case_name}-t2f.nii.gz"
        ]
        
        # Check for segmentation file (from original validation set)
        seg_file = f"{case_name}-seg.nii.gz"
        
        missing_files = []
        for req_file in required_files:
            if not os.path.exists(os.path.join(case_dir, req_file)):
                missing_files.append(req_file)
        
        if missing_files:
            print(f"Skipping {case_name}: missing files {missing_files}")
            continue
        
        try:
            # Copy modality files to nnUNet format
            shutil.copy(
                os.path.join(case_dir, f"{case_name}-t1n.nii.gz"), 
                os.path.join(imagestr, f"{case_name}_0000.nii.gz")
            )
            shutil.copy(
                os.path.join(case_dir, f"{case_name}-t1c.nii.gz"),
                os.path.join(imagestr, f"{case_name}_0001.nii.gz") 
            )
            shutil.copy(
                os.path.join(case_dir, f"{case_name}-t2w.nii.gz"),
                os.path.join(imagestr, f"{case_name}_0002.nii.gz")
            )
            shutil.copy(
                os.path.join(case_dir, f"{case_name}-t2f.nii.gz"),
                os.path.join(imagestr, f"{case_name}_0003.nii.gz")
            )
            
            # Copy and convert segmentation if it exists
            seg_path = os.path.join(case_dir, seg_file)
            if os.path.exists(seg_path):
                copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
                    seg_path,
                    os.path.join(labelstr, f"{case_name}.nii.gz")
                )
            else:
                print(f"Warning: No segmentation file found for {case_name}")
            
            converted_cases += 1
            
        except Exception as e:
            print(f"Error converting {case_name}: {e}")
            continue
    
    print(f"Successfully converted {converted_cases} cases")
    
    # Generate dataset.json
    generate_dataset_json(output_base_dir, converted_cases)
    
    return output_base_dir

def generate_dataset_json(output_base_dir, num_cases):
    """Generate dataset.json file for nnUNet."""
    
    dataset_json = {
        "channel_names": {
            "0": "T1",
            "1": "T1ce", 
            "2": "T2",
            "3": "Flair"
        },
        "labels": {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3], 
            "enhancing tumor": [3]
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "regions_class_order": [1, 2, 3],
        "license": "see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",
        "reference": "BraTS Challenge - Synthesized Missing Modalities",
        "dataset_release": "1.0"
    }
    
    import json
    with open(os.path.join(output_base_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"Generated dataset.json with {num_cases} cases")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert completed dataset to nnUNet format")
    parser.add_argument("--input_dir", default="./datasets/BRATS2023/pseudo_validation_completed",
                       help="Directory containing completed pseudo-validation data")
    parser.add_argument("--output_dir", default="./Dataset137_BraTS21_Completed", 
                       help="Output directory for nnUNet format dataset")
    
    args = parser.parse_args()
    
    # Convert to nnUNet format
    output_dir = setup_nnunet_dataset(args.input_dir, args.output_dir)
    
    print(f"\n✅ Dataset conversion complete!")
    print(f"nnUNet dataset saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Set nnUNet environment variables")
    print(f"2. Run: nnUNetv2_predict -i '{output_dir}/imagesTr' -o './outputs' -d 137 -c 3d_fullres -f 5")
    print(f"3. Calculate Dice scores: python cal_avg_dice.py")

if __name__ == "__main__":
    main()