"""
Loader for the full stack
low-res -> importance map -> sampling -> high-res reconstruction
for both isosurface and dvr.
To allow for an efficient parallel data-loading, the crops are precomputed
"""

import numpy as np
import torch
import os
import h5py
import scipy

from dataset.datasetUtils import getCropsForDataset

def getSamplePatternCrops(
    sample_pattern,
    num_crops : int,
    crop_size : int):
    """
    Returns random crops of the sampling pattern.

    Args:
     - sample_pattern: the sample pattern as a 2D numpy array
     - num_crops: the number of crops to extract
     - crop:size: the size of the crops

    Returns a numpy array of shape N*3 with the three columns being
     - x, y of the crop
     - augmentation
    """
    crops = np.zeros((num_crops, 3), dtype=np.int32)
    crops[:,0] = np.random.randint(0, sample_pattern.shape[0] - crop_size, size=num_crops, dtype=crops.dtype)
    crops[:,1] = np.random.randint(0, sample_pattern.shape[1] - crop_size, size=num_crops, dtype=crops.dtype)
    crops[:,2] = np.random.randint(-4, 4, size=num_crops, dtype=crops.dtype)
    return crops

class AdaptiveDataset(torch.utils.data.Dataset):

    def __init__(self, *, 
                 crop_size : int, 
                 dataset_file : str, 
                 dataset_name : str = 'gt',
                 dataset_low : str = None,
                 data_crops,
                 pattern,
                 pattern_crops,
                 downsampling : int):
        """
        crop_size: the size of the crops
        dataset_file: the filename of the hdf5 file with the dataset
        dataset_name: the name of the dataset inside the hdf5 file
        dataset_low: the filename of the hdf5 file with the low-resolution inputs,
           If None, the high-resolution images are downsampled
        data_crops: the numpy array with the crop definition as created by 
           getCropsForDataset(dataset_file, dataset_name, crop_size=crop_size)
        pattern: the sample pattern, a 2D numpy array
        pattern_crops: the numpy array with the crop definition as created by
           getSamplePatternCrops(pattern, crop_size=crop_size)
        downsampling: the factor to which the input
           should be downsamples to create the network input.
           'crop_size' must be a multiply of 'downsampling'

        The dataloader can be parallelized over multiple workers.
        
        Returns: a tuple of (low-resolution, high-res pattern, high-res gt)
        """
        assert crop_size == (crop_size // downsampling) * downsampling, \
            "crop_size is not a multiply of downsampling"

        self.crop_size = crop_size
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.dataset_low = dataset_low
        self.data_crops = data_crops
        self.pattern = pattern
        self.pattern_crops = pattern_crops
        self.downsampling = downsampling

        self.num_items = data_crops.shape[0]
        assert data_crops.shape[0]==pattern_crops.shape[0]

        self.hdf5_file = None
        self.dset = None

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        
        if self.hdf5_file is None:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else -1
            #print("Worker %d: load dataset"%worker_id)
            self.hdf5_file = h5py.File(self.dataset_file, 'r')
            self.dset = self.hdf5_file[self.dataset_name]
            if self.dataset_low is not None:
                self.hdf5_low_file = h5py.File(self.dataset_low, 'r')
                self.dset_low = self.hdf5_low_file[self.dataset_name]
            #print("Worker %d: loaded"%worker_id)

        # create high resolution crop
        crop_def = self.data_crops[idx,:]
        crop_high = self.dset[
            crop_def[0],                            # sample
            :,                                      # time
            :,                 # channel
            crop_def[2]:crop_def[2]+self.crop_size, # y / height
            crop_def[1]:crop_def[1]+self.crop_size] # x / width
        crop_high = torch.from_numpy(crop_high)

        # create low-resolution crop
        if self.dataset_low is None:
            crop_low = torch.nn.functional.interpolate(
                crop_high, scale_factor=1/self.downsampling,
                mode='area')
        else:
            assert self.dset_low.shape[-1] * self.downsampling == self.dset.shape[-1], \
                "You specified a low-resolution dataset, but the downsampling factor does not match:\n"+\
                "  low-res size: %d, high-res size: %d, factor: %d"%(self.dset_low.shape[-1], self.dset.shape[-1], self.downsampling)
            crop_low = self.dset_low[
                crop_def[0],                            # sample
                :,                                      # time
                :,                 # channel
                crop_def[2]//self.downsampling : (crop_def[2]+self.crop_size)//self.downsampling, # y / height
                crop_def[1]//self.downsampling : (crop_def[1]+self.crop_size)//self.downsampling] # x / width
            crop_low = torch.from_numpy(crop_low)

        # create pattern crop
        crop_def = self.pattern_crops[idx,:]
        pattern = self.cropSamplePattern(crop_def)
        pattern = np.ascontiguousarray(pattern)
        pattern = torch.from_numpy(pattern)

        return crop_low, pattern, crop_high

    def cropSamplePattern(self, crop):
        """
        Crops the sample pattern 'pattern' by 'crop',
        a row of the array that is returned by getSamplePatternCrops
        """
    
        x, y, aug = crop[0], crop[1], crop[2]
        c = self.pattern[y:y+self.crop_size, x:x+self.crop_size]
        if aug < 0:
            c = np.fliplr(c)
            aug = np.abs(aug)
        c = np.rot90(c, aug)
        return c