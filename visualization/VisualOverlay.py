"""
    VISUALOVERLAY
    Visualization of the original image and corresponding mask. Show image with overlaying mask, or save masked .nii.gz file.
    Done per slice or per volume.

    @author: tdoekemeijer
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path_to_nii_file =  '/.../PADDED/CR/Images/filename.nii.gz'
path_to_mask = '/.../PADDED/CR/Masks/filename-label.nii.gz'
print(nib.load(path_to_nii_file).header)

# load image and mask
img_data = nib.load(path_to_nii_file).get_fdata()
mask_data = nib.load(path_to_mask).get_fdata()

# perform masking and save
masked_file = img_data * mask_data
out = nib.Nifti1Image(masked_file, affine=nib.load(path_to_nii_file).affine)
# nib.save(out, '.../masked.nii.gz')

# Visualization one slice
slice = 8
figOr = img_data[:,:,slice]
figMa = mask_data[:,:,slice]
figOrMa = masked_file[:,:,slice]

f = plt.figure(figsize=(10, 8))
f.add_subplot(3,1,1) #rows, columns, position
plt.imshow(np.rot90(figOr), cmap="gray")
plt.title("Image")
f.add_subplot(3,1, 2)
plt.imshow(np.rot90(figMa), cmap="gray")
plt.title("Mask")
f.add_subplot(3,1,3)
plt.imshow(np.rot90(figOr), cmap="gray")
plt.imshow(np.rot90(figMa), cmap='jet', alpha=0.3)
plt.title("Overlay")
plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.tight_layout()
plt.show(block=True)

# Visualization whole volume
f = plt.figure(figsize=(10, 8))
for i in range(18):
    f.add_subplot(6,3, i+1) #rows, columns, position
    plt.imshow(np.rot90(img_data[:,:,i]), cmap="gray")
plt.tight_layout()
plt.show()

# Visualization all masks
f = plt.figure(figsize=(10, 8))
for i in range(18):
    f.add_subplot(6,3, i+1) #rows, columns, position
    plt.imshow(np.rot90(mask_data[:,:,i]), cmap="gray")
plt.tight_layout()
plt.show()



