# This script create a pseudo validation set during the validation stage and a test set during the final evaluation.

import os
import random
import numpy as np
import shutil

def create_pseudo_validation():
    # the target validation set foler and the new folder, please replace it with yours
    # Adjusted val_set_folder to point to the directory containing patient IDs
    val_set_base_folder = './datasets/BRATS2023/validation'
    val_set_actual_data = os.path.join(val_set_base_folder, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData')
    val_set_missing = './datasets/BRATS2023/pseudo_validation'
    val_set_missing_actual_data = os.path.join(val_set_missing, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData')


    if not os.path.exists(val_set_missing):
        os.mkdir(val_set_missing)
    if not os.path.exists(val_set_missing_actual_data):
        os.mkdir(val_set_missing_actual_data)


    # create a pseudo validation set by randomly dropping one modality
    np.random.seed(123456)  # fix random seed
    modality_list = ['t1c', 't1n', 't2f', 't2w']  # the list of modalities in the given folder

    # Now, folder_list will contain the patient IDs (e.g., BraTS-GLI-00001-000)
    patient_folder_list = os.listdir(val_set_actual_data)
    patient_folder_list.sort()

    print(f"Processing {len(patient_folder_list)} cases...")
    print(f"Creating pseudo validation set: {val_set_missing_actual_data}")


    drop_index = np.random.randint(0, 4, size=len(patient_folder_list))

    for count, patient_id_folder in enumerate(patient_folder_list):
        src_patient_path = os.path.join(val_set_actual_data, patient_id_folder)
        dst_patient_path = os.path.join(val_set_missing_actual_data, patient_id_folder)

        if not os.path.exists(dst_patient_path):
            os.mkdir(dst_patient_path)

        files_in_patient_folder = os.listdir(src_patient_path)
        
        modality_to_drop = modality_list[drop_index[count]]
        print(f"Case {patient_id_folder}: dropping {modality_to_drop}")

        for filename in files_in_patient_folder:
            # Check if the filename contains the modality to drop
            if modality_to_drop not in filename:
                src_file = os.path.join(src_patient_path, filename)
                dst_file = os.path.join(dst_patient_path, filename)
                shutil.copyfile(src_file, dst_file)

if __name__ == '__main__':
    create_pseudo_validation()