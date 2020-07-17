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

class StepsizeDataset(torch.utils.data.Dataset):

    def __init__(self, *, 
                 crop_size : int, 
                 dataset_target : str, 
                 dataset_input : str,
                 data_crops):
        """
        crop_size: the size of the crops
        dataset_target: the filename of the hdf5 file with the the target dataset.
           Must contain entries with name "step%d".
        dataset_input: the filename of the hdf5 file with the (possibly) low-resolution inputs.
           Must contain an entry with the name "gt".
           The difference in resolution between dataset_target and dataset_input
            specifies the importance map upscaling.
        data_crops: the numpy array with the crop definition as created by 
           getCropsForDataset(dataset_file, dataset_name, crop_size=crop_size)

        The dataloader can be parallelized over multiple workers.
        
        Returns: a tuple of (input, targets[]) where the first target has the lowest stepsize
        """

        self.crop_size = crop_size
        self.dataset_target = dataset_target
        self.dataset_input = dataset_input
        self.data_crops = data_crops

        self.num_items = data_crops.shape[0]

        self._hdf5_file_target = None
        self._hdf5_file_input = None
        self._dsets_target = None
        self._dset_input = None

        # parse target dataset for available stepsizes
        with h5py.File(self.dataset_target, 'r') as f:
            keys = list(f.keys())
            stepsizes = []
            for key in keys:
                if key.startswith('step'):
                    stepsizes.append(int(key[4:]))
            stepsizes.sort()
            self._min_stepsize = stepsizes[0]
            self._max_stepsize = stepsizes[-1]
            print("Stepsizes found:", stepsizes)
            self._target_keys = ["step%d"%s for s in stepsizes]
            # test that the stepsizes are a sequence of integers without gaps
            if stepsizes[-1] - stepsizes[0] != len(stepsizes)-1:
                raise ValueError("There are gaps in the step sizes!!")
            # save shape
            self._target_shape = f["step%d"%stepsizes[0]].shape

        # parse input shape for the sr-factor
        with h5py.File(self.dataset_input, 'r') as f:
            self._input_shape = f['gt'].shape

        self._upscale_factor = self._target_shape[-1] // self._input_shape[-1]
        if self._target_shape[-1] != self._input_shape[-1] * self._upscale_factor:
            raise ValueError("target resolution is not a integer-multiple of the input resolution")
        print("Upscale factor:", self._upscale_factor)

    def min_stepsize(self):
        return self._min_stepsize

    def max_stepsize(self):
        return self._max_stepsize

    def upscale_factor(self):
        return self._upscale_factor

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        
        if self._hdf5_file_target is None:
            self._hdf5_file_target = h5py.File(self.dataset_target, 'r')
            self._dsets_target = [self._hdf5_file_target[k] for k in self._target_keys]
        if self._hdf5_file_input is None:
            self._hdf5_file_input = h5py.File(self.dataset_input, 'r')
            self._dset_input = self._hdf5_file_input['gt']

        crop_def = self.data_crops[idx,:]

        # create input crop
        crop_input = torch.from_numpy(self._dset_input[
            crop_def[0],       # sample
            :,                 # time
            :,                 # channel
            crop_def[2]//self._upscale_factor : (crop_def[2]+self.crop_size)//self._upscale_factor, # y / height
            crop_def[1]//self._upscale_factor : (crop_def[1]+self.crop_size)//self._upscale_factor # x / width
            ])

        # create target crops
        crops_target = [
            torch.from_numpy(dset[
                crop_def[0],                            # sample
                :,                                      # time
                :,                 # channel
                crop_def[2]:crop_def[2]+self.crop_size, # y / height
                crop_def[1]:crop_def[1]+self.crop_size]) # x / width
            for dset in self._dsets_target
            ]
        
        return crop_input, crops_target
