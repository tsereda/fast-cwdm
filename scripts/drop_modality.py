"""
Create a pseudo validation set by randomly dropping one modality per case.
Adapted from BraSyn tutorial for fast-cwdm project structure.
"""

import os
import random
import numpy as np
import shutil

def create_pseudo_validation():
    # Paths for your project structure
    val_set_folder = 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    val_set_missing = 'pseudo_validation'
    
    if not os.path.exists(val_set_missing):
        os.makedirs(val_set_missing)
    
    # Create a pseudo validation set by randomly dropping one modality
    np.random.seed(123456)  # Fix random seed for reproducibility
    modality_list = ['t1c', 't1n', 't2f', 't2w']  # Available modalities
    
    folder_list = os.listdir(val_set_folder)
    folder_list.sort()
    
    # Randomly assign which modality to drop for each case
    drop_index = np.random.randint(0, 4, size=len(folder_list))
    
    print(f"Processing {len(folder_list)} cases...")
    print(f"Creating pseudo validation set: {val_set_missing}")
    
    for count, case_folder in enumerate(folder_list):
        case_src = os.path.join(val_set_folder, case_folder)
        case_dst = os.path.join(val_set_missing, case_folder)
        
        if not os.path.isdir(case_src):
            continue
            
        if not os.path.exists(case_dst):
            os.makedirs(case_dst)
        
        file_list = os.listdir(case_src)
        dropped_modality = modality_list[drop_index[count]]
        
        print(f"Case {case_folder}: dropping {dropped_modality}")
        
        # Copy all files except the dropped modality
        for filename in file_list:
            if dropped_modality not in filename:  # Keep files that don't contain dropped modality
                src_file = os.path.join(case_src, filename)
                dst_file = os.path.join(case_dst, filename)
                shutil.copyfile(src_file, dst_file)
                
        # Create a marker file to indicate which modality is missing
        marker_file = os.path.join(case_dst, f"missing_{dropped_modality}.txt")
        with open(marker_file, 'w') as f:
            f.write(f"Missing modality: {dropped_modality}\n")
    
    print(f"Pseudo validation set created successfully!")
    print(f"Total cases processed: {len(folder_list)}")

if __name__ == "__main__":
    create_pseudo_validation()