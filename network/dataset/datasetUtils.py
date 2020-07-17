"""
Utilities for computing crops and normalizing images
"""

import numpy as np
import torch
import os
import h5py
import scipy
from typing import List

def getCropsForDataset(
    dataset_file : str, dataset_name : str, 
    num_crops : int,
    crop_size : int, offset_factor : int,
    fill_rate_percent : int, mask_channel : int):
    """
    Returns the list of crops for the given dataset file.
    If the list does not already exist, it is created (costly) and cached.
    The dataset entries must be of shape (Batch,Time,Channels,Height,Width)

    Args:
     - dataset_file: the hdf5 file with the dataset
     - dataset_name: the name of the dataset in the hdf5 file
     - num_crops: the number of crops to create
     - crop_size: the size of the crops
     - offset_factor: ensures that the crops have coordinates that are a
        multiplication of this value
     - fill_rate_percent: the fill rate in percent of how many
        pixels must be set
     - mask_channel: the channel to check the fill rate

    Output format: a numpy array of shape N*3 where the three columns are
     - index into the dataset
     - x, y of the 
    This means, a crop 'idx' can be taken from the dataset 'dset' using
        index, x, y = crops[idx,0], crops[idx,1], crops[idx,2]
        crop = dset[index, :, :, y:y+crop_size, x:x+crop_size]
    """

    crops = None

    crop_filename = dataset_file[:-5]+"_crops.hdf5"
    crop_file = h5py.File(crop_filename, 'a')

    crop_dset_name = "crops_n%d_s%d_f%d_p%d_c%d" % (
        num_crops, crop_size, offset_factor, fill_rate_percent, mask_channel)
    
    if crop_dset_name in crop_file.keys():
        # crops already exist, just load them
        print("Load crops from cache")
        crops = crop_file[crop_dset_name][...]
    else:
        print("Crops not yet created, compute them")
        from console_progressbar import ProgressBar
        crops = np.zeros((num_crops, 3), dtype=np.int32)
        fill_rate = crop_size*crop_size*fill_rate_percent/100.0

        with h5py.File(dataset_file, "r") as f:
            dset = f[dataset_name]
            B, T, C, H, W = dset.shape
            assert crop_size <= H
            assert crop_size <= W
            pg = ProgressBar(num_crops, 'Find Crops', length=50)
            crop_index = 0
            while crop_index < num_crops:
                pg.print_progress_bar(crop_index)
                # sample possible crop
                index = np.random.randint(0, B)
                x = np.random.randint(0, W - crop_size) if crop_size < W else 0
                y = np.random.randint(0, H - crop_size) if crop_size < H else 0
                x = (x // offset_factor) * offset_factor
                y = (y // offset_factor) * offset_factor
                # check if filled
                mask = (dset[index, 0, mask_channel, y:y+crop_size, x:x+crop_size] > 0.2) * 1.0
                if np.sum(np.abs(mask)) >= fill_rate:
                    # crop found
                    crops[crop_index, 0] = index
                    crops[crop_index, 1] = x
                    crops[crop_index, 2] = y
                    crop_index += 1

            pg.print_progress_bar(num_crops)
            np.random.shuffle(crops)
        print("Save crops to cache")

        crop_dset = crop_file.create_dataset(crop_dset_name, data=crops)
        crop_dset.attrs["num_crops"] = num_crops
        crop_dset.attrs["crop_size"] = crop_size
        crop_dset.attrs["offset_factor"] = offset_factor
        crop_dset.attrs["fill_rate_percent"] = fill_rate_percent
        crop_dset.attrs["mask_channel"] = mask_channel
        crop_file.flush()

    crop_file.close()
    return crops

class Normalization:

    def __init__(self, channels : List[int], means : List[float], stds : List[float]):
        """
        Creates the normalization tool.
        The processed input tensors are expected to be of shape B*C*H*W

        channels: list of integers which channels are affected
        mean:s list of means for the specified channels
        stds: list of variances of the specified channels

        If channels is empty, the normalizations are no-ops

        The normalization operation performs logically the following operation,
        except for not being in-place to support gradients:
            
            output = input.clone()
            for channel, mean, std in zip(channels, means, stds):
                output[channel] = (output[channel] - mean) / std

        """

        assert len(channels)==len(means)
        assert len(channels)==len(stds)

        self._channels = channels
        self._means = means
        self._stds = stds

    def getParameters(self):
        """
        Returns the normalization parameters as a dict with the keys 'channels', 'means', 'stds'
        """
        return {'channels':self._channels, 'means':self._means, 'stds':self._stds}

    class Normalize(torch.nn.Module):
        def __init__(self, channels, means, stds):
            super().__init__()
            self._channels = channels
            self._means = means
            self._stds = stds
        def forward(self, input):
            if len(self._channels)==0: return input
            cx = list(torch.split(input, 1, dim=1))
            for channel, mean, std in zip(self._channels, self._means, self._stds):
                cx[channel] = (cx[channel] - mean) / std
            return torch.cat(cx, dim=1)

    def getNormalize(self):
        return Normalization.Normalize(self._channels, self._means, self._stds)

    class Denormalize(torch.nn.Module):
        def __init__(self, channels, means, stds):
            super().__init__()
            self._channels = channels
            self._means = means
            self._stds = stds
        def forward(self, input):
            if len(self._channels)==0: return input
            cx = list(torch.split(input, 1, dim=1))
            for channel, mean, std in zip(self._channels, self._means, self._stds):
                cx[channel] = (cx[channel] * std) + mean
            return torch.cat(cx, dim=1)

    def getDenormalize(self):
        return Normalization.Denormalize(self._channels, self._means, self._stds)

def getNormalizationForDataset(
    dataset_file : str, dataset_name : str, 
    channels : List[int]):
    """
    Returns the normalization for the given dataset file.
    If the normalization settings don't already exist, it is created (costly) and cached.
    The settings are cached in the same '_crops.hdf5' file as the crops.

    The dataset entries must be of shape (Batch,Time,Channels,Height,Width)

    If channels is empty, the normalizations are no-ops

    Args:
     - dataset_file: the hdf5 file with the dataset
     - dataset_name: the name of the dataset in the hdf5 file
     - channels: the list of channels considered for normalization

    Output: an instance of Normalization
    """

    means : List[float] = [None] * len(channels)
    stds : List[float] = [None] * len(channels)

    crop_filename = dataset_file[:-5]+"_crops.hdf5"
    crop_file = h5py.File(crop_filename, 'a')

    norm_dset_name = "norm-%s" % ('-'.join(map(str, channels)))
    
    if norm_dset_name in crop_file.keys():
        # crops already exist, just load them
        data = crop_file[norm_dset_name][...]
        assert data.shape==(2, len(channels)), "illegal shape of the normalization cache, expected %s, got %s"%((2, len(channels)), data.shape)
        means = list(data[0])
        stds = list(data[1])
        print("Load normalization from cache: channels=%s, means=%s, stds=%s"%(channels, means, stds))
    else:
        print("Normalization not yet created, compute them")
        from console_progressbar import ProgressBar
        from utils.mv import MeanVariance
        from math import sqrt

        mvsX = [MeanVariance() for i in range(len(channels))]
        mvsX2 = [MeanVariance() for i in range(len(channels))]

        with h5py.File(dataset_file, "r") as f:
            dset = f[dataset_name]
            B, T, C, H, W = dset.shape
            pg = ProgressBar(B*T, 'Compute Statistics', length=40)
            for b in range(B):
                for t in range(T):
                    pg.print_progress_bar(t + T*b)
                    img = dset[b, t, ...]
                    img2 = img * img
                    for idx, c in enumerate(channels):
                        mvsX[idx].append(np.mean(img[c,:,:]))
                        mvsX2[idx].append(np.mean(img2[c,:,:]))
            pg.print_progress_bar(B*T)
            for idx in range(len(channels)):
                means[idx] = mvsX[idx].mean()
                stds[idx] = sqrt(mvsX2[idx].mean() - (mvsX[idx].mean()**2))
        print("Computed normalization: channels=%s, means=%s, stds=%s. Save to cache"%(channels, means, stds))

        # save
        data = np.stack([means, stds], axis=0)
        assert data.shape==(2, len(channels))
        norm_dset = crop_file.create_dataset(norm_dset_name, data=data)
        crop_file.flush()

    crop_file.close()
    return Normalization(channels, means, stds)