# Automatic Segmentation of Pancreatic Tumors
This repository includes the code for my research project: Deep Learning for Automatic Segmentation of Pancreatic Tumors in Diffusion-Weighted MR Images. Below, you can find a summary of the project and an overview of the repository. 

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
