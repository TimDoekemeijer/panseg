"""
    NRRD_NIIGZ
    Convert .nrrd files to .nii.gz files. Done per .nii.gz file.

    @author: tdoekemeijer
"""

import nrrd
import nibabel as nib
import numpy as np

baseDir = '/.../Primair_labelmap/'
file = 'filename.nrrd'
nrrd = nrrd.read(baseDir+file)
data = nrrd[0]
header = nrrd[1]

#save nifti
img = nib.Nifti1Image(data, np.eye(4))
last_chars = file[-24:-5]
nib.save(img,baseDir+last_chars+'.nii.gz')
