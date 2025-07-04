# "Fast-"cWDM: Conditional Wavelet Diffusion Models for Cross-Modality 3D Medical Image Synthesis

### "The Intersection of Learnable Timestep Scheduling and Wavelet-Based Diffusion Models: A Literature Review" Conclusion

"The intersection of learnable timestep scheduling and wavelet-based diffusion models represents a significant untapped research opportunity with substantial theoretical and practical implications. The literature reveals robust individual advances in both domains, established theoretical foundations for integration, and compelling evidence of benefits in medical imaging applications. However, the specific combination remains unexplored, presenting a clear opportunity for novel contributions.

The most promising immediate direction involves developing **Frequency-Adaptive Timestep Scheduling (FATS)**, which could leverage wavelet energy statistics to learn optimal noise schedules for different frequency bands. This approach addresses current limitations in both domains while opening new possibilities for more efficient, controllable, and theoretically grounded diffusion models. The computational feasibility is strong, with existing wavelet implementations and established scheduling optimization techniques providing a solid foundation.

This research direction has the potential to significantly advance generative modeling by providing more efficient and principled approaches that better exploit the natural hierarchical structure of visual data, with particular promise for medical imaging applications where both computational efficiency and generation quality are paramount."

## Paper Abstract

## Dependencies
We recommend using a [conda](https://github.com/conda-forge/miniforge#mambaforge) environment to install the required dependencies.
You can create and activate such an environment called `fast-cwdm` by running the following commands:
```sh
mamba env create -f environment.yml
mamba activate fast-cwdm
```

## Training & Sampling
For training a new model or sampling from an already trained one, you can simply adapt and use the script `run.sh`. All relevant hyperparameters for reproducing our results are automatically set when using the correct `MODEL` in the general settings.
For executing the script, simply use the following command:
```sh
bash run.sh
```

## Pre-trained model weights
We release pre-trained model weights on [HuggingFace](https://huggingface.co/).

Simply download the weights and replace the path to the model weights in the `sample_auto.py` script.

## Data
The provided code works for the following data structure (you might need to adapt the `DATA_DIR` variable in `run.sh`):
```
data
└───BRATS
    └───training
        └───BraTS-GLI-00000-000
            └───BraTS-GLI-00000-000-seg.nii.gz
            └───BraTS-GLI-00000-000-t1c.nii.gz
            └───BraTS-GLI-00000-000-t1n.nii.gz
            └───BraTS-GLI-00000-000-t2f.nii.gz
            └───BraTS-GLI-00000-000-t2w.nii.gz  
        └───BraTS-GLI-00002-000
        ...

    └───validation
        └───BraTS-GLI-00001-000
            └───BraTS-GLI-00001-000-t1c.nii.gz
            └───BraTS-GLI-00001-000-t1n.nii.gz
            └───BraTS-GLI-00001-000-t2f.nii.gz
            └───BraTS-GLI-00001-000-t2w.nii.gz  
        └───BraTS-GLI-00001-001
        ...       
```

## Acknowledgements
Our code is based on / inspired by the following repositories:
* [https://github.com/pfriedri/cwdm](https://github.com/pfriedri/cwdm) (published under MIT License)
