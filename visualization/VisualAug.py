'''
    VISUALAUG
    Visualization of the augmented images.
    Done per slice.

    @author: tdoekemeijer
'''

from train.utils import *
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import DataLoader
import warnings
import torchvision
warnings.filterwarnings('ignore')


def vis_preprocessing(filename_pair):
    transforms = torchvision.transforms.Compose([
        HistogramClipping(),
        RangeNorm(),
        # only for training
        mt_transforms.RandomRotation(rot_degree),  # random rotation
        mt_transforms.RandomAffine(0, translate=transl_range),  # shift
        mt_transforms.RandomAffine(0, shear=shear_range),  # shear

        mt_transforms.ToTensor(),
    ])

    dataset = MRI2DSegmentationDataset([filename_pair],
                                       transform=transforms,
                                       #slice_filter_fn=slice_filtering,
                                       )

    dataloader = DataLoader(dataset, batch_size=18, collate_fn=mt_datasets.mt_collate)

    return dataloader

#augmentations
rot_degree = 10
transl_range = [0.05, 0.05]
shear_range = [-5, 5]


# Get filename
nifti_image = '/.../PADDED/CR/Images/filename.nii.gz'
nifti_mask = '/.../PADDED/CR/Masks/filename-label.nii.gz'
filename_pair = nifti_image, nifti_mask
print(filename_pair)

dataloader = vis_preprocessing(filename_pair)
batch = next(iter(dataloader))

# Visualize one slice
slice = 7
input_slice = batch['input'][slice].squeeze(0)
input_slice = ndimage.rotate(input_slice, 90)
gt_slice = batch['gt'][slice].squeeze(0)
gt_slice = ndimage.rotate(gt_slice, 90)

f = plt.figure(figsize=(12, 6))
f.add_subplot(2, 1, 1)  # rows, columns, position
plt.imshow(input_slice, cmap="gray")
plt.title("Image")
# plt.axis("off")
plt.colorbar(shrink=0.65)
plt.grid(True)
f.add_subplot(2, 1, 2)
plt.imshow(gt_slice, cmap="gray")
plt.title("Mask")
# plt.axis("off")
plt.colorbar(shrink=0.65)
plt.grid(True)
plt.tight_layout()
plt.show()
