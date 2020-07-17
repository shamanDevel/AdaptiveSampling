"""
A little script that takes the importance-sampled inputs,
samples and inpaints them and saves them as input for a dense training.
This allows to train the dense reconstruction network without the sampling process.
"""

import numpy as np
import torch
import h5py
import os
from typing import Tuple, List
from collections import namedtuple
from enum import Enum
from contextlib import ExitStack
import time
import progressbar

import models
import importance

class Inpainting(Enum):
    FAST = 1
    PDE = 2

Settings = namedtuple("Settings", ['pattern', 'importanceMin', 'importanceMean', 'inpainting', 'output_name'])

BORDER_CROP = 16

class EnhanceNetWrapper(torch.nn.Module):
    def __init__(self, orig):
        super().__init__();
        self._orig = orig
    def forward(self, input, mask):
        return self._orig(input)[0]

def convert(
    dataset_high : str, # path to the high-resolution dataset
    dataset_low : str,  # path to the low-resolution dataset
    pattern : str, # path to the dataset with the sample pattern
    network : str,      # path to the importance network
    output_prefix : str,# prefix of the output datasets
    settings : List[Settings], # configurations to compute
    mode : str # 'ISO' or 'DVR'
    ):

    device = torch.device('cuda')

    print("Load Renderer.dll")
    torch.ops.load_library("./Renderer.dll")
    print()

    # load datasets
    dataset_high_file = h5py.File(dataset_high, 'r')
    dataset_low_file = h5py.File(dataset_low, 'r')
    dset_high = dataset_high_file['gt']
    dset_low = dataset_low_file['gt']
    B, T, C, Hhigh, Whigh = dset_high.shape
    _, _, _, Hlow, Wlow = dset_low.shape
    upscaling_factor = Hhigh // Hlow

    print("number of samples:", B)
    print("with timesteps: ", T)
    print("a resolution of: C=%d, H=%d, W=%d"%(C, Hhigh, Whigh))
    print("and an upscaling factor of: ", upscaling_factor)
    print()

    # load pattern
    pattern_file = h5py.File(pattern, 'r')
    print("available sampling pattern: ", pattern_file.keys())
    patterns = dict()
    pattern_min_size = 1<<20
    for key in pattern_file.keys():
        patterns[key] = torch.from_numpy(pattern_file[key][...]).unsqueeze(0).to(device)
        print("Pattern", key, "loaded, shape:", patterns[key].shape)
        pattern_min_size = min(pattern_min_size, patterns[key].shape[0], patterns[key].shape[1])
    print()

    # load network
    print("Load network")
    checkpoint = torch.load(network)
    importanceModel = checkpoint['importanceModel']
    importanceModel.to(device)
    print("Network loaded")
    parameters = checkpoint['parameters']
    importanceNetUpscale = parameters['importanceNetUpscale']
    print("network upscaling:", importanceNetUpscale)
    if importanceNetUpscale > upscaling_factor:
        print("Network upscaling is larger than the the dataset upscaling. Terminate")
        return
    importancePostUpscale = upscaling_factor // importanceNetUpscale
    print("post upscaling:", importancePostUpscale)
    if mode == 'ISO':
        channels_low = 5 # mask, normal x, normal y, normal z, depth
    elif mode == 'DVR':
        channels_low = 8 # rgba, normal xyz, depth
    else:
        raise ValueError("Unknown mode '%s', 'ISO' or 'DVR' expected"%mode)
    print()

    # create postprocess for each setting
    importancePostprocess = [
        importance.PostProcess(
            s.importanceMin, s.importanceMean, 
            importancePostUpscale,
            parameters['lossBorderPadding'] // importancePostUpscale,
            'basic')
        for s in settings]

    # open outputs
    with torch.no_grad():
        with ExitStack() as stack:
            output_files = [
                stack.enter_context(h5py.File(output_prefix + s.output_name, 'w'))
                for s in settings]
            dset_high_cropped_shape = tuple(
                list(dset_high.shape)[:-2] +
                [dset_high.shape[-2]-2*BORDER_CROP, dset_high.shape[-1]-2*BORDER_CROP])
            output_dsets = [
                f.create_dataset("gt", dset_high_cropped_shape, dset_high.dtype)
                for f in output_files]
            for ds in output_dsets:
                ds.attrs["Mode"] =  mode
            print("Output files created")
            print("Process datasets now...")

            # loop over the dataset
            widgets=[
                ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
                progressbar.Counter(),
                ' (', progressbar.ETA(), ') ',
            ]
            for b in progressbar.progressbar(range(B), widgets=widgets):
            
                # get the time slice
                data_high = torch.from_numpy(dset_high[b, ...]).to(device)
                data_low = torch.from_numpy(dset_low[b, ...]).to(device)

                # evaluate the network (time becomes batch)
                importance_input = data_low[:, :channels_low, :, :]
                previous_input = torch.zeros(
                    T,1,
                    importance_input.shape[2]*importanceNetUpscale,
                    importance_input.shape[3]*importanceNetUpscale, 
                    dtype=data_low.dtype, device=device)
                importance_input = torch.cat([
                    importance_input,
                    models.VideoTools.flatten_high(previous_input, importanceNetUpscale)
                    ], dim=1)
                importance_map = importanceModel(importance_input)

                # get pattern crop
                sampling_pattern_x = np.random.randint(0, pattern_min_size - Hhigh) if Hhigh < pattern_min_size else 0
                sampling_pattern_y = np.random.randint(0, pattern_min_size - Whigh) if Whigh < pattern_min_size else 0

                # loop over all settings
                for sIdx in range(len(settings)):
                    s = settings[sIdx]

                    # get the pattern
                    sampling_pattern = patterns[s.pattern][:, sampling_pattern_x:sampling_pattern_x+Hhigh, sampling_pattern_y:sampling_pattern_y+Whigh]

                    # normalize and perform the sampling
                    importance_map_post, _ = importancePostprocess[sIdx](importance_map)
                    sample_mask = (importance_map_post >= sampling_pattern).to(dtype=importance_map_post.dtype).unsqueeze(1)

                    # run the inpainting
                    crop = sample_mask * data_high
                    if s.inpainting == Inpainting.FAST:
                        inpainted = importance.fractionalInpaint(crop, sample_mask[:,0,:,:])
                    elif s.inpainting == Inpainting.PDE:
                        inpainted = importance.pdeInpaint(crop, sample_mask[:,0:1,:,:], cpu=False)
                    else:
                        assert False, "unknown inpainting algorithm"

                    # save the output
                    output_dsets[sIdx][b,...] = inpainted.cpu().numpy() \
                        [..., BORDER_CROP:-BORDER_CROP, BORDER_CROP:-BORDER_CROP]

def repairCrop(folder, input_filename, output_filename, mode):
    """
    Previous versions of convert(...) didn't account for border cropping,
    so crop now.
    """
    dataset_in_file = h5py.File(folder + input_filename, 'r')
    dset_in = dataset_in_file['gt']
    B, T, C, H, W = dset_in.shape

    dset_out_shape = (B, T, C, H - 2*BORDER_CROP, W - 2*BORDER_CROP)
    with h5py.File(folder + output_filename, 'w') as dataset_out_file:
        dset_out = dataset_out_file.create_dataset(
            "gt", dset_out_shape, dset_in.dtype)
        dset_out.attrs["Mode"] = mode
        dset_out[...] = dset_in[..., BORDER_CROP:-BORDER_CROP, BORDER_CROP:-BORDER_CROP]


if __name__ == "__main__":
    settings = [
        #Settings("halton", 0.002, 0.05, Inpainting.FAST, "halton_05_fast.hdf5"),
        #Settings("halton", 0.002, 0.05, Inpainting.PDE,  "halton_05_pde.hdf5"),
        #Settings("halton", 0.001, 0.01, Inpainting.FAST, "halton_01_fast.hdf5"),
        #Settings("halton", 0.001, 0.01, Inpainting.PDE,  "halton_01_pde.hdf5"),
        Settings("plastic", 0.002, 0.10, Inpainting.FAST, "plastic_10_fast.hdf5"),
        #Settings("plastic", 0.002, 0.05, Inpainting.PDE,  "plastic_10_pde.hdf5"),
        Settings("plastic", 0.002, 0.10, Inpainting.FAST, "plastic_05_fast.hdf5"),
        #Settings("plastic", 0.002, 0.05, Inpainting.PDE,  "plastic_05_pde.hdf5"),
        Settings("plastic", 0.001, 0.01, Inpainting.FAST, "plastic_01_fast.hdf5"),
        #Settings("plastic", 0.001, 0.01, Inpainting.PDE,  "plastic_01_pde.hdf5"),
        ]

    if 0:
        dataset_high = "D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2.hdf5"
        dataset_low = "D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-screen8x.hdf5"
        pattern = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        network = "D:/VolumeSuperResolution/adaptive-modeldir/imp050/model_epoch_500.pth"
        output_prefix = "D:/VolumeSuperResolution-InputData/gt-inpainted-"
        mode = 'ISO'
    elif 1:
        dataset_high = "D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6.hdf5"
        dataset_low = "D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6-screen8x.hdf5"
        pattern = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        network = "D:/VolumeSuperResolution/adaptive-dvr-modeldir/adapDvr5-rgb-temp001-perc01/model_epoch_300.pth"
        output_prefix = "D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6-"
        mode = 'DVR'

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    #convert(dataset_high, dataset_low, pattern, network, output_prefix, settings, mode)

    repairCrop("D:/VolumeSuperResolution-InputData/",
               "gt-dvr-ejecta6.hdf5",
               "gt-dvr-ejecta6-inpaintCrop.hdf5",
               "DVR")
    #for s in settings:
    #    input_filename = s.output_name[:-5] + "_old.hdf5"
    #    output_filename = s.output_name
    #    print("Process", input_filename)
    #    repairCrop(output_prefix, input_filename, output_filename)