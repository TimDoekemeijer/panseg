'''
    VISUALIMBALANCE
    Visualization of background imbalance.

    @author: tdoekemeijer
'''

from train.utils import *
from misc.getLists import *
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

dirName = '/.../PADDED_CROP/'
listOfImages = getListOfPPImages(dirName)
listOfMasks = getListOfPPMasks(dirName)
filename_pairs = get_filename_pairs(listOfImages, listOfMasks)

# Print all pairs
for x,y in filename_pairs:
    Im = nib.load(x)
    Mask = nib.load(y)
    ImSize = Im.shape
    MaSize = Mask.shape
    print(x, ImSize, y, MaSize)
print("\nNumber of volumes:", len(filename_pairs), "\n")

def visualize(filename_pairs):
    dataset = MRI2DSegmentationDataset(filename_pairs,
                                       # slice_filter_fn=slice_filtering_count,
                                       )

    dataloader = DataLoader(dataset, batch_size=2161, collate_fn=mt_datasets.mt_collate)

    return dataloader

dataloader = visualize(filename_pairs)
batch = next(iter(dataloader))
print("Number of slices:", len(batch['input']))

# Start stack of zero-padded sized images
total_pixels = 0
total_tum = 0
if dirName == '/.../PADDED/':
    stacked_image = np.zeros((160,64))
    stacked_mask = np.zeros((160,64))
if dirName == '/.../PADDED_CROP/':
    stacked_image = np.zeros((128, 32))
    stacked_mask = np.zeros((128,32))

# Stack images and masks
for i in range(len(batch['input'])):
    input_slice_PIL = batch['input'][i]
    input_slice = np.array(input_slice_PIL)
    gt_slice_PIL = batch['gt'][i]
    gt_slice = np.array(gt_slice_PIL)

    num_pixels = np.size(gt_slice)
    total_pixels += num_pixels

    num_tum = np.count_nonzero(gt_slice)
    total_tum += num_tum

    stacked_image = np.add(stacked_image,input_slice)
    stacked_mask = np.add(stacked_mask, gt_slice)

# Calculate imbalance percentage
perc = (total_tum/total_pixels) * 100
print(f'Percentage of foreground pixels:{perc}')

# Plot stacked images/masks
f = plt.figure(figsize=(12, 6))
f.add_subplot(2, 1, 1)  # rows, columns, position
plt.imshow(np.rot90(stacked_image), cmap="jet")
plt.title("Stacked images")
plt.axis("off")
plt.tight_layout()
plt.colorbar()
f.add_subplot(2, 1, 2)
plt.imshow(np.rot90(stacked_mask), cmap="jet")
plt.title("Stacked labels")
plt.axis("off")
plt.tight_layout()
plt.colorbar()
plt.show()