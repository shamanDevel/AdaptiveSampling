
import torch.utils.data as data

import os.path
import collections
import random
import numpy as np
import torch
import time
import h5py
from typing import Callable, Optional, Union, List

from dataset.datasetUtils import getCropsForDataset

"""
Data-Format:

Let W, H be the width and height of the high-resolution images that are sampled.
Let the downsampling factor be 4, for example.
The low resolution images will then have a resolution of W/4 and H/4.

Let T be the number of timesteps per sequence.
Typically, the sequences are 10 frames long

== ISOSURFACE ==
Each sample contains three numpy arrays
 - high, the target of the network of shape T*C*H*W
   with C=6 and the channels are defined as:
    - mask (+1=intersection, -1=no intersection)
    - normalX [-1,+1],
    - normalY [-1,+1],
    - normalZ [-1,+1],
    - depth [0,1],
    - ao [0,1] where 1 means no occlusion, 0 full occlusion

 - low, the input of the network of shape T*C*(H/4)*(W/4)
   with C=5 and the channels are defined as:
    - mask (+1=intersection, -1=no intersection)
    - normalX [-1,+1],
    - normalY [-1,+1],
    - normalZ [-1,+1],
    - depth [0,1],

  - flow, the low-resolution flow of shape T*2*(H/4)*(W/4)
    with the following two channels:
    - flowX [-1,+1] inpainted flow in normalized coordinates
    - flowY [-1,+1] inpainted flow in normalized coordinates

== DVR ==
Each sample contains three numpy arrays
 - high, the target of shape T*C*H*W
   with C=4 and the channels
    - red
    - green
    - blue
    - alpha

 - low, the input of the network of shape T*C*(H/4)*(W/4)
   with C=8 and the channels
    - red
    - green
    - blue
    - alpha
    - normal x
    - normal y
    - normal z
    - depth

 - flow, the low-resolution flow of shape T*2*(H/4)*(W/4)
   with the following two channels:
    - flowX [-1,+1] inpainted flow in normalized coordinates
    - flowY [-1,+1] inpainted flow in normalized coordinates
   Note that DVR rendering currently can't compute flow,
   hence this field is always filled with zeros

"""

_dvr_data_agumentation_shuffle = [
        (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
def dvr_data_augmentation(index, mode, data, _shuffle=_dvr_data_agumentation_shuffle):
    if mode == "high" or mode == "low":
        s = _shuffle[index % len(_shuffle)]
        data = np.concatenate((
            data[:,s[0]:s[0]+1,:,:],
            data[:,s[1]:s[1]+1,:,:],
            data[:,s[2]:s[2]+1,:,:],
            data[:,3:,:,:]
            ), axis=1)
    return data

class DenseDatasetFromSamples_v2(data.Dataset):
    """
    Output of __getitem__ (in that order):
     - images_low: num_frames x input_channels x crop_size x crop_size
     - flow_low: num_frames x 2 x crop_size x crop_size
     - images_high: num_frames x output_channels x crop_size*upsampling x crop_size*upsampling
    """
    def __init__(self, 
                 target_dataset_path : str,
                 input_dataset_path : str,
                 input_channels : Union[int, List[int]], 
                 output_channels : int,
                 num_crops : int,
                 crop_size : int, # high-res crop size, if -1 then the full-res images are used
                 test : bool, #True: test set, False: training set
                 test_fraction : float, #Fraction of images used for the test set
                 upscaling : int, # upscaling factor for validation of the datasets
                 fill_rate_percent : float,
                 augmentation : Optional[Callable[[int, str, np.ndarray], np.ndarray]] = None):
        super().__init__()
        self._target_dataset_path = target_dataset_path
        self._input_dataset_path = input_dataset_path
        if isinstance(input_channels, int):
            self._input_channels = list(range(input_channels))
        elif isinstance(input_channels, list):
            self._input_channels = input_channels
        else:
            raise TypeError("unknown input_channel type %s (%s)"%(input_channels, type(input_channels)))
        self._output_channels = output_channels
        self._upscaling = upscaling
        self._augmentation = augmentation

        # preload dataset to get shapes
        with h5py.File(target_dataset_path, 'r') as targetf,\
             h5py.File(input_dataset_path, 'r') as inputf:
            target = targetf['gt']
            input = inputf['gt']
            self.channels_high = target.shape[2]
            self.channels_low = input.shape[2]
            assert self.channels_high >= self._output_channels
            assert self.channels_low >= len(self._input_channels)
            self._size_high = target.shape[3]
            self._size_low = input.shape[3]
            actual_upscaling = self._size_high // self._size_low
            if actual_upscaling != upscaling:
                raise ValueError("expected an upscaling of %d, but the datasets imply a scaling of %d"%(
                    upscaling, actual_upscaling))

        # compute crops
        if crop_size == -1:
            crop_size = self._size_high
        if crop_size % upscaling != 0:
            raise ValueError("crop size of %d is not a multiple of the upscaling factor %d"%(
                crop_size, upscaling))
        self._crops = getCropsForDataset(
            target_dataset_path, 'gt',
            num_crops, crop_size, upscaling,
            fill_rate_percent, 0)

        self._crop_size_low = crop_size // upscaling
        self._crop_size_high = crop_size
        self._flow_scaling = self._size_low / self._crop_size_low
        self.num_images = self._crops.shape[0]

        l = int(self.num_images * test_fraction)
        if test:
            self.index_offset = self.num_images - l
            self.num_images = l
        else:
            self.index_offset = 0
            self.num_images -= l

        # the final datasets are loaded per process
        self.hdf5_file_high = None
        self.hdf5_file_low = None
        self.dset_high = None
        self.dset_low = None

    def get_low_res_shape(self, channels):
        return (channels, self._crop_size_low, self._crop_size_low)

    def get_high_res_shape(self):
        return (self._output_channels, self._crop_size_high, self._crop_size_high)

    def get(self, index, mode):
        idx = self._crops[index+self.index_offset, 0]
        x = self._crops[index+self.index_offset, 1]
        y = self._crops[index+self.index_offset, 2]
        if mode=='high':
            d = self.dset_high[
                idx, 
                :, :self._output_channels, 
                y : y + self._crop_size_high, 
                x : x + self._crop_size_high]
        elif mode=='low':
            d = self.dset_low[
                idx, 
                :, self._input_channels, 
                y//self._upscaling : (y//self._upscaling) + self._crop_size_low, 
                x//self._upscaling : (x//self._upscaling) + self._crop_size_low]
        elif mode=='flow':
            d = self._flow_scaling * self.dset_high[
                idx, 
                :, -2:, 
                y : y + self._crop_size_high : self._upscaling, 
                x : x + self._crop_size_high : self._upscaling]
        if self._augmentation is not None:
            d = self._augmentation(index, mode, d)
        return d

    def __getitem__(self, index):
        # load hdf5 files
        if self.hdf5_file_high is None:
            self.hdf5_file_high = h5py.File(self._target_dataset_path, 'r')
            self.dset_high = self.hdf5_file_high['gt']
            self.hdf5_file_low = h5py.File(self._input_dataset_path, 'r')
            self.dset_low = self.hdf5_file_low['gt']

        # get items
        low = torch.from_numpy(self.get(index, 'low'))
        high = torch.from_numpy(self.get(index, 'high'))
        flow = torch.from_numpy(self.get(index, 'flow'))
        return low, flow, high

    def __len__(self):
        return self.num_images

