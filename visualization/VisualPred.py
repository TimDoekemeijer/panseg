'''
    VISUALPRED
    Visualization of the original image, mask, and predicted mask. Overlay in edges.
    Done per slice or per volume.

    @author: tdoekemeijer
'''

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

path_to_nii_file =  '/.../PADDED/CR/Images/filename.nii.gz'
path_to_mask = '/.../PADDED/CR/Masks/filename-label.nii.gz'
path_to_pred = '/.../SLURMoutput/prednif/Prediction1.nii.gz'

# Load image and mask and prediction
img_data = nib.load(path_to_nii_file).get_fdata()
print(img_data.shape)
mask_data = nib.load(path_to_mask).get_fdata()
print(mask_data.shape)
pred_data = nib.load(path_to_pred).get_fdata()
print(pred_data.shape)

masked_file = img_data * -mask_data

slice = 8
figOr = img_data[:,:,slice+3]
figMa = mask_data[:,:,slice+3]
figPr = pred_data[:,:,slice]

img = cv2.merge((figOr,figOr,figOr))

# Only take edges out of prediction areas
edgedcopy = np.uint8(figPr)
edged = cv2.Canny(edgedcopy, threshold1=1, threshold2=0)
img2 = img.copy()
img2[edged == 255] = [0, 255, 0]  # turn edges to green - Prediction

edgedcopy2 = np.uint8(figMa)
edged2 = cv2.Canny(edgedcopy2, threshold1=1, threshold2=0)
img3 = img.copy()
img3[edged2 == 255] = [255, 0, 0]  # turn edges to red - Ground truth

figOrMa = masked_file[:,:,slice]

f = plt.figure(figsize=(20, 10))
f.add_subplot(2,1,1) #rows, columns, position
plt.imshow(np.rot90(figOr), cmap="gray")
plt.title("Image")
plt.axis("off")
f.add_subplot(2,1, 2)
plt.imshow(np.rot90(figOr), cmap="gray")
plt.imshow(np.rot90(img3), cmap="jet", alpha=.2)
plt.imshow(np.rot90(img2), cmap="jet", alpha=.2)
plt.title("Predicted Tumor in green vs. Ground Truth in red")
plt.axis("off")
plt.tight_layout()
plt.show(block=True)

# Show image slices
f = plt.figure(figsize=(10, 8))
for i in range(18):
    f.add_subplot(6,3, i+1) #rows, columns, position
    plt.imshow(np.rot90(img_data[:,:,i]), cmap="gray")
plt.tight_layout()
plt.show()

# Show mask slices
f = plt.figure(figsize=(10, 8))
for i in range(18):
    f.add_subplot(6,3, i+1) #rows, columns, position
    plt.imshow(np.rot90(mask_data[:,:,i]), cmap="gray")
plt.tight_layout()
plt.show()


# Show mask without background slices
# Change for each mask
mask_data_np = mask_data[:, :, 0:18] #which slices are foreground #van, tot not including
print(mask_data_np.shape)

f = plt.figure(figsize=(10, 8))
for i in range(18):
    f.add_subplot(6,3, i+1) #rows, columns, position
    plt.imshow(np.rot90(mask_data_np[:, :, i]), cmap="gray")
plt.tight_layout()
plt.show()

# Show predictions
f = plt.figure(figsize=(10, 8))
for i in range(18):
    f.add_subplot(6,3, i+1) #rows, columns, position
    plt.imshow(np.rot90(pred_data[:,:,i]), cmap="gray")
plt.tight_layout()
plt.show()

# Show overlap
f = plt.figure(figsize=(5, 10))
for i in range(18):
    f.add_subplot(9,2, i+1) #rows, columns, position
    plt.imshow(np.rot90(mask_data_np[:,:,i]), cmap="gray")
    plt.imshow(np.rot90(pred_data[:, :, i]), cmap='jet', alpha=0.3)
    plt.axis("off")
    plt.tight_layout()
plt.show()
