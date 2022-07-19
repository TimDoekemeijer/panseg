"""
    CENTERCROPPING
    Crop a list of images or masks to a specific size and save them as .nii.gz.
    Only takes primary tumor, done per study

    @author: tdoekemeijer
"""

from misc.getLists import *
import nibabel as nib
import numpy as np

dirName = '/.../STUDIES/MATRIX/'
dirPadIm = '/.../PADDED_CROP/MATRIX/Images/'
dirPadMa = '/.../PADDED_CROP/MATRIX/Masks/'
size_x = 128
size_y = 32
size_z = 18

def crop_center(img,cropx,cropy):
    x,y,z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[startx:startx+cropx, starty:starty+cropy, :]

listOfZPCRImages, listOfZPCRMasks = getListOfPPImages(dirName), getListOfPPMasks(dirName)
listOfZPREPROImages, listOfZPREPROMasks = getListOfREPROImages(dirName), getListOfREPROMasks(dirName)
listOfZPMATRIXImages, listOfZPMATRIXMasks = getListOfMATRIXImages(dirName), getListOfMATRIXMasks(dirName)

# Original dataset with sizes to be converted
filename_pairs = get_filename_pairs(listOfZPMATRIXImages, listOfZPMATRIXMasks)
for x,y in filename_pairs:
    Im = nib.load(x)
    Mask = nib.load(y)
    ImSize = Im.shape
    MaSize = Mask.shape
    print(x, ImSize, y, MaSize)
print("\nNumber of pairs:", len(filename_pairs), "\n")

# Crop images
for item in listOfZPMATRIXImages:
    img = nib.load(item)
    img_array = img.get_fdata()
    cropped_image = crop_center(img_array, size_x, size_y)
    print(item, cropped_image.shape)

    ni_img = nib.Nifti1Image(cropped_image, img.affine)
    last_chars = item[52:]                                  #for CR48, REPRO51, MATRIX52
    last_chars2 = last_chars.replace('/','_')
    nib.save(ni_img, dirPadIm + last_chars2)

# Crop masks
# For masks, also only returns the primary tumors and disregards the other ones.
for item in listOfZPCRMasks:
    msk = nib.load(item)
    msk_array = msk.get_fdata()
    msk_prim = np.where(msk_array < 2, msk_array, 0)        #takes only primary tumor
    cropped_mask = crop_center(msk_prim, size_x, size_y)

    ni_img = nib.Nifti1Image(cropped_mask, msk.affine)
    last_chars = item[51:]                                  #for CR47, REPRO50, MATRIX51
    last_chars2 = last_chars.replace('/','_')
    nib.save(ni_img, dirPadMa + last_chars2)

