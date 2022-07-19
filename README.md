# Automatic Segmentation of Pancreatic Tumors
This repository includes the code for my research project: Deep Learning for Automatic Segmentation of Pancreatic Tumors in Diffusion-Weighted MR Images. Below, you can find a summary of the project and an overview of the repository. 

## Summary 
Pancreatic cancer has an exceptionally high mortality rate, making it one of the most common causes of cancer mortality in developed countries. By quantitatively studying the microenvironment of these pancreatic tumors, parameters can be extracted, which can help with diagnosis. An important diagnostic imaging modality to evaluate these tumors is Intravoxel Incoherent Motion (IVIM) MRI, an extension of diffusion-weighted imaging (DWI). Retrieving the quantitative parameters from IVIM MR images necessitates contouring regions of interest (ROIs), a process that is time-consuming, labor-intensive, and prone to observer variation. The aim of this project was to research the use of CNNs, specifically U-Net, to automatically and accurately contour pancreatic tumors on IVIM MR images. The used dataset was retrieved from three separate studies, and consisted of 61 patients, with a total of 119 IVIM MRI volumes. Each volume consisted of 18 axial slices, from which a total of 767 slices were labeled. Leave one out cross-validation (LOOCV) was used to most efficiently split the dataset into training, validation, and testing portions. Multiple loss functions were explored, and hyperparameters of the model were optimized using an automatic software framework for ML, monitoring the mean Dice validation loss. Preprocessing was used to make the dimensions of volumes of equal size and enhance contrast, and data augmentation (rotating, shifting, and shearing) was used to make the model more generalizable. First, experiments were conducted with only foreground slices, after which varying amounts of background slices were added to avoid bias towards predictions on every slice. Additionally, experiments with cropping were applied to counter class imbalance. Model performance was evaluated on a never-before seen part of the dataset, using the Dice Similarity Coefficient (DSC) metric. Algorithms trained and evaluated on foreground slices yield the highest mean DSC with 0.39 (+- 0.22) for zero-padded input images, and 0.40 (+- 0.22) for cropped images, comparable to the mean DSC of anatomical scans in literature, but showing great variance in DSCs between patients. The algorithms trained, validated, and evaluated on input data with multiple added background slices, performed significantly worse than algorithms trained, validated, and evaluated on only foreground slices. In conclusion, U-net is not able to accurately predict pancreatic tumors on IVIM MR volumes from this dataset. 

## Overview
### Preliminaries 
To be able to use this code, the added conda environment (pansegenvironment.yml) needs to be installed.

### Contents
The contents of this repository are as follows:
- **conversion**: Contains scripts to convert files to NIfTI, and to zero-pad or crop volumes to the appropriate size. 
- **misc**: Contains a script to make lists of input volumes, and the hyperparameter optimizion script.
- **train**: Contains the code for training, including a utils scripts with helper functions, a seperate script with loss functions, and a shell script.
- **visualization**: Contains different scripts to visualize the dataset, the class imbalance, and the predictions of the model.

### Usage
The training process can be run with SLURM, using the shell script:
```
sbatch panseg.sh
```
