"""
    ZEROPADDING
    Zero-pad a list of images or masks to a specific size and save them as .nii.gz.
    Only takes primary tumor, done per study

    @author: tdoekemeijer
"""

from misc.getLists import *
import nibabel as nib
import numpy as np

dirName = '/.../STUDIES/MATRIX/'
dirPadIm = '/.../PADDED/MATRIX/Images/'
dirPadMa = '/.../PADDED/MATRIX/Masks/'
size_x = 160
size_y = 64
size_z = 18

def padding(array, xx, yy, zz):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :param zz: desired depth
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    d = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz - d) // 2
    cc = zz - c - d

    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')

listOfCRImages, listOfCRMasks = getListOfCRImages(dirName), getListOfCRMasks(dirName)
listOfREPROImages, listOfREPROMasks = getListOfREPROImages(dirName), getListOfREPROMasks(dirName)
listOfMATRIXImages, listOfMATRIXMasks = getListOfMATRIXImages(dirName), getListOfMATRIXMasks(dirName)

#Original dataset with sizes to be converted
filename_pairs = get_filename_pairs(listOfMATRIXImages, listOfMATRIXMasks)
for x,y in filename_pairs:
    Im = nib.load(x)
    Mask = nib.load(y)
    ImSize = Im.shape
    MaSize = Mask.shape
    print(x, ImSize, y, MaSize)
print("\nNumber of pairs:", len(filename_pairs), "\n")

# Zeropad images
for item in listOfMATRIXImages:
    img = nib.load(item)
    img_array = img.get_fdata()
    padded_image = padding(img_array, size_x, size_y, size_z)

    ni_img = nib.Nifti1Image(padded_image, img.affine)
    last_chars = item[39:]
    last_chars2 = last_chars.replace('/','_')
    nib.save(ni_img, dirPadIm + last_chars2)

# Zeropad masks
#For masks, also only returns the primary tumors and disregards the other ones.
for item in listOfMATRIXMasks:
    msk = nib.load(item)
    msk_array = msk.get_fdata()
    msk_prim = np.where(msk_array < 2, msk_array, 0)            #takes only primary tumor
    padded_mask = padding(msk_prim, size_x, size_y, size_z)

    ni_img = nib.Nifti1Image(padded_mask, msk.affine)
    last_chars = item[39:]
    last_chars2 = last_chars.replace('/','_')
    nib.save(ni_img, dirPadMa + last_chars2)