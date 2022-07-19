"""
    UTILS
    Functions and classes needed to execute the training script for automatic pancreatic tumor segmentation.

    @author: tdoekemeijer
"""

import random
import numpy as np
from PIL import Image
from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
from torch.utils.data import Dataset
import nibabel as nib
from tqdm import tqdm

### FUNCTIONS ###
countslice = 0
countlab = 0
def slice_filtering_count0(slice_pair):
    ''' These slice filtering techniques disregard slices that do not contain
    a mask with a certain number -> countlab < 0 passes only foreground slices.
    countlab >= 0 passes whole volumes. Works only with 18 axial slices'''
    global countslice
    global countlab
    countslice += 1
    if countslice == 18:
        countslice, countlab = 0,0

    if len(np.unique(slice_pair['gt'])) == 2:
        return True
    elif len(np.unique(slice_pair['gt'])) == 1 and countslice > 1 and countlab < 0:
        countlab += 1
        return True
    elif countlab >= 2:
        return False

def slice_filtering_count2(slice_pair):
    ''' These slice filtering techniques disregard slices that do not contain
    a mask with a certain number -> ratio foreground/background 3/1: countlab < 2'''
    global countslice
    global countlab
    countslice += 1
    if countslice == 18:
        countslice, countlab = 0,0

    if len(np.unique(slice_pair['gt'])) == 2:
        return True
    elif len(np.unique(slice_pair['gt'])) == 1 and countslice > 1 and countlab < 2:
        countlab += 1
        return True
    elif countlab >= 2:
        return False

def slice_filtering_count5(slice_pair):
    ''' These slice filtering techniques disregard slices that do not contain
    a mask with a certain number -> passes 5 background slices each volume'''
    global countslice
    global countlab
    countslice += 1
    if countslice == 18:
        countslice, countlab = 0,0

    if len(np.unique(slice_pair['gt'])) == 2:
        return True
    elif len(np.unique(slice_pair['gt'])) == 1 and countslice > 0 and countlab < 5:
        countlab += 1
        return True
    elif countlab >= 2:
        return False

def sub_filename_pairs(uniques, train_index, test_index, listpat, filename_pairs):
    ''' Makes filename pairs for train and test after k-fold split'''
    pat_test = []
    pat_train = []
    for index in test_index:
        pat_test.append(uniques[index])
    for index in train_index:
        pat_train.append(uniques[index])

    print(f"Patients in test: {pat_test}, total {len(pat_test)}")

    blist = []
    for entry in listpat:
        if entry in pat_test:
            blist.append(True)
        if entry in pat_train:
            blist.append(False)

    test = []
    train = []
    for i in range(len(blist)):
        if blist[i] == True:
            test.append(filename_pairs[i])
            random.shuffle(test)
        if blist[i] == False:
            train.append(filename_pairs[i])
            random.shuffle(train)

    return test, train

def reset_weights(m):
    ''' Resets model weights, used for k-fold cross validation'''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


### CLASSES ###
class HistogramClipping(mt_transforms.MTTransform):
    ''' Performs histogram clipping. '''

    def __init__(self, min_percentile=2.0, max_percentile=98.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        np_input_data = np.array(input_data)

        mask = np.where(np_input_data > 0)
        masked = np_input_data[mask].flatten()

        percentile1 = np.nanpercentile(masked, self.min_percentile)  # np.percentile(masked, self.min_percentile)
        percentile2 = np.nanpercentile(masked, self.max_percentile)  # np.percentile(masked, self.max_percentile)

        np_input_data[np_input_data <= percentile1] = percentile1
        np_input_data[np_input_data >= percentile2] = percentile2

        input_data = Image.fromarray(np_input_data, mode='F')
        rdict['input'] = input_data

        sample.update(rdict)
        return sample

class RangeNorm(mt_transforms.MTTransform):
    ''' Performs range/intensity normalization. '''

    def __init__(self, minv=0, maxv=255):
        self.min = minv
        self.max = maxv

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        np_input_data = np.array(input_data)

        maxv = np.max(np_input_data)

        np_input_data = (np_input_data / maxv) * 255

        input_data = Image.fromarray(np_input_data, mode='F')
        rdict['input'] = input_data

        sample.update(rdict)
        return sample

class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).
    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """

    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            # Changed: instead of throwing a warning, change the dimension
            # of the input
            self.input_handle = nib.funcs.four_to_three(self.input_handle)[0]

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).
        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = mt_datasets.SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = mt_datasets.SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn

class MRI2DSegmentationDataset(Dataset):
    """
    - From medicaltorch.datasets.MRI2DSegmentationDataset()
    This is a generic class for 2D (slice-wise) segmentation datasets.
    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """

    def __init__(self, filename_pairs, slice_axis=2, p=.5, cache=True,
                 transform=None, preprocess=None, slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.preprocess = preprocess
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical
        self.p = p

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SegmentationPair2D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[2]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.
        :param transform: the new transformation
        """
        self.transform = transform

    def compute_mean_std(self, verbose=False):
        """Compute the mean and standard deviation of the entire dataset.
        :param verbose: if True, it will show a progress bar.
        :returns: tuple (mean, std dev)
        """
        sum_intensities = 0.0
        numel = 0

        with mt_datasets.DatasetManager(self,
                                        override_transform=mt_transforms.ToTensor()) as dset:
            pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_intensities += input_data.sum()
                numel += input_data.numel()
                pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                                 refresh=False)

            training_mean = sum_intensities / numel

            sum_var = 0.0
            numel = 0

            pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_var += (input_data - training_mean).pow(2).sum()
                numel += input_data.numel()
                pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                                 refresh=False)

        training_std = np.sqrt(sum_var / numel)
        return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).
        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(pair_slice["input"], mode='F')

        #REMOVED # Handle unlabeled data
        gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'gt': gt_img,
            'input_metadata': pair_slice['input_metadata'],
            'gt_metadata': pair_slice['gt_metadata'],
        }

        # Added: augment only a probability p of the data
        if self.transform is not None:
            if np.random.random() < self.p:
                data_dict = self.transform(data_dict)
            else:
                if self.preprocess is not None:
                    data_dict = self.preprocess(data_dict)
        else:
            if self.preprocess is not None:
                data_dict = self.preprocess(data_dict)

        return data_dict
