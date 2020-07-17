"""
Trains the full pipeline of
low-resolution input -> importance.NetworkImportanceMap -> sampling -> sparse reconstruction network -> high resolution image
Sadly, it has a ton of parameters
"""

from __future__ import print_function
import argparse
import math
from math import log10
import os
import os.path
from collections import defaultdict, namedtuple
import itertools
import shutil
import sys
import traceback
import matplotlib.pyplot as plt
import subprocess
import logging
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import h5py

no_summary = False
try:
    from torchsummary import summary
except ModuleNotFoundError:
    no_summary = True
    print("No summary writer found")

from console_progressbar import ProgressBar

import dataset.adaptiveDatasetLoader
import models
import losses
import importance
from utils import ScreenSpaceShading, initialImage, getColorSpaceConversions, MeanVariance

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='Full-pipeline adaptive sampling training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    class ActionStoreDeprecated(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            logging.warn('DEPRECATED: using %s', '|'.join(self.option_strings))
            setattr(namespace, self.dest, values)

    parser.add_argument('--mode', type=str, choices=['iso', 'dvr'], help="""
        The operation mode of the networks, possible options are:
        'iso': unshaded isosurface superresolution.
            The dataset contains images of eight channels in the order mask, normalX,-Y,-Z, depth, ao, flowX, flowY
            The input to the network is mask, normalXYZ, depth
            The final reconstruction network gets as input the sampled mask, normalXYZ, depth and the sampling mask in the last channel
            The final reconstruction network outputs mask, normalXYZ, depth, ao.
        'dvr': superresolution on shaded direct volume rendering
            The dataset contains images of four channels: red, green, blue, alpha
            TODO: define network inputs, outputs, what happens with flow?
        """)

    #Dataset
    parser_group = parser.add_argument_group("Dataset")
    parser_group.add_argument('--dataset', type=str, required=True,
                        help="Path the the HDF5 file with the dataset of the high-resolution image")
    parser_group.add_argument('--datasetLow', type=str, default=None, help="""
        Path the the HDF5 file with the dataset of the low-resolution image.
        This is optional, if not specified, the high-resolution images will be downscaled.""")
    parser_group.add_argument('--samples', type=int, required=True, help='Number of samples for the train and test dataset')
    parser_group.add_argument('--cropSize', type=int, required=True, help="The size of the crops")
    parser_group.add_argument('--cropFillRate', type=int, default=40, help="By how much must each crop be filled in percent")
    parser_group.add_argument('--testFraction', type=float, default=0.2, help='Fraction of test data')
    parser_group.add_argument('--numWorkers', type=int, default=0, help="Number of dataloader")
    parser_group.add_argument('--inputChannels', type=str, default=None, help="""
        Selects which channels from the input dataset are used.
        Isosurface, a comma-separated list of
         - normal
         - depth
         (mask is always implied)
        DVR, a comma-separated list of
         - color (rgba)
         - normal
         - depth
        The default values, if this argument is ommitted, are:
         - Isosurface: "normal,depth"
         - DVR: "color"
    """)
    parser_group.add_argument('--colorSpace', type=str, default='rgb',
                              choices=['rgb', 'hsv', 'xyz', 'luv'], help="""
        Specifies in which color space the network should process color channels.
        Currently supported are: rgb, hsv, xyz, luv (case-insensitive).
        This option only applies to DVR images.""")
    parser_group.add_argument('--normalize', type=str, default="",  help="""
        Specifies the channels which should be normalized over the whole dataset before sending to the network
        as a comma-separated list. Default: no normalization / empty list.""")

    #Restore
    parser_group = parser.add_argument_group("Restore")
    parser_group.add_argument('--restore', type=str, default=None, help="Restore training from the specified run index")
    parser_group.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
    parser_group.add_argument('--pretrainedImportance', type=str, default=None, 
                              help="Path to a pretrained importance-map network")
    parser_group.add_argument('--pretrainedReconstruction', type=str, default=None, 
                              help="Path to a pretrained importance-map network")

    #Importance Model parameters
    parser_group = parser.add_argument_group("Importance-Model")
    parser_group.add_argument('--importanceModel', type=str, default="EnhanceNet", choices=['EnhanceNet'], help="""
    The model for importance map generation, support is only 'EnhanceNet' so far.""")
    parser_group.add_argument('--importanceNetUpscale', type=int, default=4, help="upscale factor of the importance map network")
    parser_group.add_argument('--importanceLayers', type=int, default=5, help="Number of layers in the importance map network")
    parser_group.add_argument('--importanceUseBN', action='store_true', help='Enable batch normalization in the generator and discriminator')
    parser_group.add_argument('--importanceBorder', type=int, default=8, help='Zero border around the network')
    parser_group.add_argument('--importancePadding', type=str, default='zero', choices=['zero', 'partial'],
                              help="Padding mode of the network, either 'zero' or 'partial'")
    parser_group.add_argument('--importanceOutputLayer', type=str, 
                              default='softplus', choices=['none', 'softplus', 'sigmoid'], help="""
        Network output layer, either 'none', 'softplus' or 'sigmoid'.
        DEPRECATED: use importanceNormalization instead.""",
        action=ActionStoreDeprecated)
    parser_group.add_argument('--importanceNormalization', type=str,
                              default='basic', choices=['basic', 'softplus', 'sigmoid'], help="""
        Specifies the normalization of the importance map. Possible options are:
         - basic: importance output in [0,\infty), requires importanceOutputLayer=softplus
         - softplus: importance output in [0,\infty), uses a softplus beforehand,
         - sigmoid: importance output in [0,1],
        all with mean specified with --importanceMean and minimal value
        specified with --importanceMin.""")
    parser_group.add_argument('--importanceMin', type=float, default=0.01,
                              help="the minimal importance value, i.e. the maximal spacing of samples")
    parser_group.add_argument('--importanceMean', type=float, default=0.2,
                              help="the mean importance value, i.e. the average number of samples")
    parser_group.add_argument('--importanceResidual', type=str, default='off', choices=['off', 'gradient'], help="""
        Uses a residual network for the importance network, the network learns changes to some baseline method.
        Possible values are:
        - off: no residual architecture
        - gradient: screen-space gradient (aka curvature) is used as baseline method
        """)
    parser_group.add_argument('--importancePostUpscale', type=int, default=1, 
                        help="Upscaling factor applied after the network for a smoother map")
    parser_group.add_argument('--importanceDisableTemporal', action='store_true', help='Disables temporal consistency')
    parser_group.add_argument('--importanceDontTrain', action='store_true',
                              help="Disables training of the importance sampler")

    #Sample Pattern parameters
    parser_group = parser.add_argument_group("Sample-Pattern")
    parser_group.add_argument('--pattern', type=str, required=True, help="""
        format: 'filename,name' where
        filename is the path to the HDF5 file with the sampling pattern and
        name is the name of the dataset inside this hdf5 file""")
    parser_group.add_argument('--sampleSharpness', type=float, default=50, help="""
        To make the pipeline differentiable, the step function in where to sample is replaced by a smooth function.
        This value specifies the steepness of that function. As it approaches infinity, the function becomes a step function again.""")

    # Reconstruction Model
    parser_group = parser.add_argument_group("Reconstruction-Model")
    parser_group.add_argument('--reconModel', type=str, default='EnhanceNet', choices=['UNet', 'DeepFovea', 'EnhanceNet'],
                                help="The network architecture, supported are 'UNet' and 'DeepFovea'")
    parser_group.add_argument('--reconLayers', type=int, default=6, help="The depth of the network")
    parser_group.add_argument('--reconFilters', type=int, default=6, help=""" 
            UNet: an int, the number of filters in the first layer of the UNet is 2**this_value.
            DeepFovea: n integers with n=depth specifying the number of features per layer.""")
    parser_group.add_argument('--reconPadding', type=str, default="zero", choices=['zero','partial'],
                                help="The padding mode, can either be 'zero' or 'partial'.")
    parser_group.add_argument('--reconUseBN', action='store_true', help="UNet: Use batch normalization in the network")
    parser_group.add_argument('--reconPartialTrainMask', action='store_true', help="""
        If partial convolutions are selected as padding, the mask that is passed through
        the convolutions and multiplied with the values. If this option is selected,
        gradients are computed for these masks and propagated back.""")
    parser_group.add_argument('--reconResidual', action='store_true', help="""
        Use residual connections from input to output.
        This option implies --reconInterpolateInput""")
    parser_group.add_argument('--reconInterpolateInput', action='store_true', help="""
        Interpolates the sparse samples before sending them to the network.
        Should most likely be used together with --reconResidual.
        This requires that the custom renderer ops are loaded.""")
    parser_group.add_argument('--reconHardInput', action='store_true', help="""
        UNet:
        If true, the valid input pixels are directly copied to the output.
        This hardly enforces that the sparse input samples are preserved in the output,
        instead of relying on the network and loss function to not change them.""")
    parser_group.add_argument('--reconUpMode', type=str, default='upsample', choices=['upconv', 'upsample'],
                                help="UNet: The upsample mode")
    parser_group.add_argument('--reconDisableTemporal', action='store_true', help='Disables temporal consistency')
    #parser_group.add_argument('--reconInitialImage', type=str, default='zero', 
    #                            choices=['zero','unshaded','input'], help="""
    #    Specifies what should be used as the previous high res frame for the first frame of the sequence,
    #    when no previous image is available from the previous predition.
    #    Available options:
    #        - zero: fill everything with zeros (default)
    #        - unshaded: Special defaults for unshaded mode: mask=-1, normal=[0,0,1], depth=0.5, ao=1
    #        - input: Use the interpolated input
    #    """)
    parser_group.add_argument("--reconDisable", action='store_true', help="""
        Disables the reconstruction network. The sparse samples are only reconstructed by the inpainting.
        This allows to test if the sampling does make sense in the first place.
        This option requires --reconInterpolateInput and excludes --importanceDontTrain.
    """)

    #Loss parameters
    parser_group = parser.add_argument_group("Loss")
    parser_group.add_argument('--losses', type=str, required=True, help="""
        comma-separated list of loss terms with weighting as string
        Format: <loss>:<target>:<weighting>
        with: loss in {l1, l2, tl2, dssim, lpips, bounded, bce, perceptual}
              target in {mask, normal, depth, color, ao, flow}
              weighting a positive number
    """)
    parser_group.add_argument('--lossesTest', type=str, default=None, help="""
        Comma-separated list of *additional* loss functions for testing: 
        l1,l2,ssim,lpips,perceptual,texture,adv. 
        Optinally, the weighting factor can be specified with a colon.
        Example: "--losses perceptual:0.1,texture:1e2,adv:10"
    """)
    parser_group.add_argument('--perceptualLossLayers', 
                        type=str, 
                         # defaults found with VGGAnalysis.py
                        default='conv_1:0.026423,conv_2:0.009285,conv_3:0.006710,conv_4:0.004898,conv_5:0.003910,conv_6:0.003956,conv_7:0.003813,conv_8:0.002968,conv_9:0.002997,conv_10:0.003631,conv_11:0.004147,conv_12:0.005765,conv_13:0.007442,conv_14:0.009666,conv_15:0.012586,conv_16:0.013377', 
                        help="""
    Comma-separated list of layer names for the perceptual loss. 
    Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
    Optinally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
    """)
    parser_group.add_argument('--textureLossLayers', type=str, default='conv_1,conv_3,conv_5', help="""
    Comma-separated list of layer names for the perceptual loss. 
    Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
    Optinally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
    """)
    parser_group.add_argument('--lossBorderPadding', type=int, default=16, help="""
    Because flow + warping can't be accurately estimated at the borders of the image,
    the border of the input images to the loss (ground truth, low res input, prediction)
    are overwritten with zeros. The size of the border is specified by this parameter.
    Pass zero to disable this padding. Default=16 as in the TecoGAN paper.
    """)
    parser_group.add_argument('--lossAO', type=float, default=1.0, help="""
    Strength of ambient occlusion in the loss function. Default=1
    """)
    parser_group.add_argument('--lossAmbient', type=float, default=0.1, help="""
    Strength of the ambient light color in the loss function's shading
    """)
    parser_group.add_argument('--lossDiffuse', type=float, default=0.9, help="""
    Strength of the diffuse light color in the loss function's shading
    """)
    parser_group.add_argument('--lossSpecular', type=float, default=0.0, help="""
    Strength of the specular light color in the loss function's shading
    """)
    parser_group.add_argument('--lossHeatmapMean', type=float, default=0.1, 
                              help="Loss weight of the additional loss term that forces the heatmap to have mean 0.5 before normalization")

    # Training
    parser_group = parser.add_argument_group("Training")
    parser_group.add_argument('--trainBatchSize', type=int, default=16, help='training batch size')
    parser_group.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
    parser_group.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser_group.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser_group.add_argument('--lrGamma', type=float, default=0.5, help='The learning rate decays every lrStep-epochs by this factor')
    parser_group.add_argument('--lrStep', type=int, default=500, help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
    parser_group.add_argument('--weightDecay', type=float, default=0, help="Weight decay (L2 penalty), if supported by the optimizer. Default=0")
    parser_group.add_argument('--optim', type=str, default="Adam", help="""
    Optimizers. Possible values: RMSprop, Rprop, Adam (default).
    """)
    parser_group.add_argument('--noCuda', action='store_true', help='Disable cuda')
    parser_group.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')

    # Output
    parser_group = parser.add_argument_group("Output")
    parser_group.add_argument('--logdir', type=str, default='D:/VolumeSuperResolution/importance_logdir', help='directory for tensorboard logs')
    parser_group.add_argument('--modeldir', type=str, default='D:/VolumeSuperResolution/importance_modeldir', help='Output directory for the checkpoints')
    parser_group.add_argument('--numVisImages', type=int, default=8, help='The number of test images that are saved for visualization')
    parser_group.add_argument('--visImagesFreq', type=int, default=10, 
                              help='checkpoints are saved every "visImagesFreq" epoch')
    parser_group.add_argument('--checkpointFreq', type=int, default=10, 
                              help='checkpoints are saved every "checkpointFreq" epoch')
    parser_group.add_argument('--useCheckpointing', action='store_true', help="""
        Uses checkpointing during training. Checkpointing saves memory, but requires more time.
        See https://pytorch.org/docs/stable/checkpoint.html for more details.""")
    parser_group.add_argument('--runName', type=str, help="""
        The name of the run in the output folder.
        If this parameter is omitted, the default name, 'run%%05d' with an increasing index is used.""")

    opt = parser.parse_args()
    opt_dict = vars(opt)
    opt_dict['type'] = 'adaptive2'
    """
    adaptive1: importance model has no temporal consistency
    adaptive2: importance model with temporal consistency
    """

    #########################
    # BASIC CONFIGURATION
    #########################

    # load renderer
    if opt.reconResidual and not opt.reconInterpolateInput:
        print("WARNING: you requested residual connections in the reconstruction network, "+
            "but you didn't specify --reconInterpolateInput as well. Is this desired?")
    if opt.reconInterpolateInput:
        print("Reconstruction network are configured with residual connections, so load the renderer now.")
        # detect suffix
        if os.name == "nt":
            rendererSuffix = ".dll"
        elif os.name == "posix":
            rendererSuffix = ".so"
        else:
            raise RuntimeError("unknown operation system "+os.name)
        # find renderer library
        rendererPath = "./Renderer" + rendererSuffix
        copyToCurrent = False
        if not os.path.isfile(rendererPath):
            rendererPath = '../bin/Renderer' + rendererSuffix
            copyToCurrent = True
        if not os.path.isfile(rendererPath):
            raise ValueError("Unable to locate Renderer" + rendererSuffix)
        rendererPath = os.path.abspath(rendererPath)
        print("renderer found at:", rendererPath)
        # for some reason, pytorch can only load libraries in the local folder
        if copyToCurrent:
            shutil.copy(rendererPath, "./")
            print("Can only load from current working directory, copy renderer to current directory")
        # load library
        rendererFile = './Renderer' + rendererSuffix
        print("attempt to load", rendererFile)
        torch.ops.load_library(rendererFile)
        print("Renderer library loaded")

    if opt.importanceDontTrain and opt.pretrainedImportance is None:
        raise ValueError("you have specified to not train the importance sampler (--importanceDontTrain), "+\
            "but you didn't specify a pretrained importance sample network via --pretrainedImportance")
    if opt.reconDisable and not opt.reconInterpolateInput:
        raise ValueError("option --reconDisable requires --reconInterpolateInput")
    if opt.importanceDontTrain and opt.reconDisable:
        raise ValueError("can't use --reconDisable and --importanceDontTrain together")

    # configurate Torch
    if not opt.noCuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --noCuda")
    device = torch.device("cpu" if opt.noCuda else "cuda")

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.set_num_threads(4)

    #########################
    # RESERVE OUTPUT DIRECTORY
    #########################

    # Run directory
    def findNextRunNumber(folder):
        files = os.listdir(folder)
        files = sorted([f for f in files if f.startswith('run')])
        if len(files)==0:
            return 0
        return int(files[-1][3:])
    nextRunNumber = max(findNextRunNumber(opt.logdir), findNextRunNumber(opt.modeldir)) + 1
    if opt.runName is not None:
        print("Use manually-defined run name '%s'"%opt.runName)
        runName = opt.runName
        logdir = os.path.join(opt.logdir, runName)
        modeldir = os.path.join(opt.modeldir, runName)
        if os.path.exists(logdir) or os.path.exists(modeldir):
            checkpoints = [f for f in os.listdir(modeldir) if f.endswith(".pth")]
            if len(checkpoints)==0:
                print("RESTARTING a previously crashed run")
                for f in os.listdir(modeldir):
                    os.remove(os.path.join(modeldir, f))
                for f in os.listdir(logdir):
                    os.remove(os.path.join(logdir, f))
            elif opt.restore is not None:
                print("run directory already exists, used for restarting")
            else:
                print("ERROR: specified run directory already exists!")
                exit(-1)
        else:
            os.makedirs(logdir)
            os.makedirs(modeldir)
    elif opt.restore == None:
        print('Current run: %05d'%nextRunNumber)
        runName = 'run%05d'%nextRunNumber
        logdir = os.path.join(opt.logdir, runName)
        modeldir = os.path.join(opt.modeldir, runName)
        os.makedirs(logdir)
        os.makedirs(modeldir)

    #########################
    # CHANNELS
    #########################

    selectedChannelsIn = []
    selectedChannelsOut = []
    residualChannels = []
    if opt.mode == 'iso':
        selectedChannelsInList = (opt.inputChannels or "normal,depth").split(',')
        selectedChannelsIn = [('mask', 0)]
        if "normal" in selectedChannelsInList:
            selectedChannelsIn += [('normalX', 1), ('normalY', 2), ('normalZ', 3)]
            residualChannels = [1,2,3]
        if "depth" in selectedChannelsInList:
            selectedChannelsIn += [('depth', 4)]
        channels_low = len(selectedChannelsIn) # mask, normal xyz, depth
        channels_sampling = channels_low + 2 # + AO (always 1), samples
        channels_out = 6 # mask, normal xyz, depth, ao
        selectedChannelsOut = list(range(6))
    elif opt.mode == 'dvr':
        selectedChannelsInList = (opt.inputChannels or "color").split(',')
        selectedChannelsIn = []
        if "color" in selectedChannelsInList:
            selectedChannelsIn += [('red', 0), ('green', 1), ('blue', 2), ('alpha', 3)]
            residualChannels = [0,1,2]
        if "normal" in selectedChannelsInList:
            selectedChannelsIn += [('normalX', 4), ('normalY', 5), ('normalZ', 6)]
        if "depth" in selectedChannelsInList:
            selectedChannelsIn += [('depth', 7)]
        channels_low = len(selectedChannelsIn)
        channels_sampling = channels_low + 1 # plus samples
        channels_out = 4 # red green blue alpha
        selectedChannelsOut = list(range(4))
    else:
        raise ValueError("unknown mode "+opt.mode)
    channels_low_with_previous = channels_low + opt.importanceNetUpscale**2
    channels_sampling_with_previous = channels_sampling + channels_out
    print("selected input channels:", selectedChannelsIn)
    selectedChannelsIn = [t[1] for t in selectedChannelsIn]


    #########################
    # DATASETS
    #########################

    print('===> Loading datasets')
    upscale_factor = opt_dict['importanceNetUpscale']*opt_dict['importancePostUpscale']

    # find crops in the images
    image_crops = dataset.adaptiveDatasetLoader.getCropsForDataset(
        opt.dataset, 'gt',
        opt.samples, opt.cropSize,
        upscale_factor, opt.cropFillRate, 
        0 if opt.mode=='iso' else 3) # mask: iso -> 0 (mask, normal, ...), dvr -> 3 (rgb, alpha)

    # load sampling pattern and find crops
    sample_pattern_name, sample_pattern_dset = opt.pattern.split(',')
    with h5py.File(sample_pattern_name, 'r') as f:
        sample_pattern = f[sample_pattern_dset][...]
    pattern_crops = dataset.adaptiveDatasetLoader.getSamplePatternCrops(
        sample_pattern, opt.samples, opt.cropSize)

    # normalization
    normalizationChannels = [int(c) for c in opt.normalize.split(',')] if opt.normalize else []
    normalizationInfo : dataset.Normalization = \
        dataset.getNormalizationForDataset(opt.dataset, 'gt', normalizationChannels)
    normalize = normalizationInfo.getNormalize()
    denormalize = normalizationInfo.getDenormalize()

    # create dataset
    l = int(opt.samples * opt.testFraction)
    train_set = dataset.adaptiveDatasetLoader.AdaptiveDataset(
        crop_size = opt.cropSize,
        dataset_file = opt.dataset, dataset_name = 'gt',
        dataset_low = opt.datasetLow,
        data_crops = image_crops[l:,:],
        pattern = sample_pattern,
        pattern_crops = pattern_crops[l:,:],
        downsampling = upscale_factor)
    test_set = dataset.adaptiveDatasetLoader.AdaptiveDataset(
        crop_size = opt.cropSize,
        dataset_file = opt.dataset, dataset_name = 'gt',
        dataset_low = opt.datasetLow,
        data_crops = image_crops[:l,:],
        pattern = sample_pattern,
        pattern_crops = pattern_crops[:l,:],
        downsampling = upscale_factor)

    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.trainBatchSize, shuffle=True, num_workers=opt.numWorkers)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.numWorkers)

    #############################
    # IMPORTANCE-MAP MODEL
    #############################
    print('===> Building importance-map model')
    importanceModel = importance.NetworkImportanceMap(
        opt.importanceNetUpscale, 
        channels_low_with_previous,
        model = opt.importanceModel,
        use_bn = opt.importanceUseBN,
        border_padding = opt.importanceBorder,
        output_layer = opt.importanceOutputLayer,
        num_layers = opt.importanceLayers,
        residual = opt.importanceResidual,
        residual_channels = residualChannels,
        padding = opt.importancePadding)
    importancePostprocess = importance.PostProcess(
            opt.importanceMin, opt.importanceMean, 
            opt.importancePostUpscale,
            0,
            opt.importanceNormalization)
            # opt.lossBorderPadding // opt.importancePostUpscale) # this leads to more border artifacts

    # apply normalization
    importanceModel = importance.NormalizedAndColorConvertedImportanceNetwork(
        importanceModel,
        normalize = normalize,
        colorSpace = opt.colorSpace if opt.mode == 'dvr' else None)

    importanceModel.to(device)
    print('Importance-Map Model:')
    print(importanceModel)
    if not no_summary:
        try:
            summary(importanceModel, 
                input_size=(channels_low_with_previous, 
                            (opt.cropSize+2*opt.importanceBorder)//upscale_factor, 
                            (opt.cropSize+2*opt.importanceBorder)//upscale_factor), 
                batch_size = 2, device=device.type)
        except RuntimeError as ex:
            print("ERROR in summary writer:", ex)

    #############################
    # RECONSTRUCTION MODEL
    #############################
    print('===> Building reconstruction model')
    class EnhanceNetWrapper(nn.Module):
        """deprecated"""
        def __init__(self, orig):
            super().__init__();
            self._orig = orig
        def forward(self, input, mask):
            return self._orig(input, mask)[0]
    if opt.reconModel == "UNet":
        reconstructionModel = models.UNet(
            in_channels = channels_sampling_with_previous, 
            out_channels = channels_out,
            depth = opt.reconLayers, 
            wf = opt.reconFilters, 
            padding = opt.reconPadding,
            batch_norm = opt.reconUseBN, 
            residual = opt.reconResidual,
            hard_input = opt.reconHardInput, 
            up_mode = opt.reconUpMode,
            return_masks = False)
    elif opt.reconModel == "EnhanceNet":
        ReconOpt = namedtuple("ReconOpt", ["useBN", "return_residual", "padding", "train_mask"])
        reconOpt = ReconOpt(
            useBN = opt.reconUseBN, return_residual=False,
            padding = opt.reconPadding, train_mask = opt.reconPartialTrainMask)
        reconstructionModel = models.EnhanceNet(
            upscale_factor = 1,
            input_channels = channels_sampling_with_previous,
            channel_mask = list(range(channels_out)),
            output_channels = channels_out,
            opt=reconOpt)
        #reconstructionModel = EnhanceNetWrapper(_reconstructionModel)
    else:
        raise Valueerror("Unsupported reconstruction model "+opt.reconModel)

    # apply normalization
    reconstructionModel = models.NormalizedAndColorConvertedNetwork(
        reconstructionModel,
        normalize = normalize, denormalize=denormalize,
        colorSpace = opt.colorSpace if opt.mode == 'dvr' else None)

    reconstructionModel.to(device)
    print('Reconstruction Model:')
    print(reconstructionModel)
    if not no_summary:
        try:
            summary(reconstructionModel, 
                input_size=[
                    (channels_sampling_with_previous, opt.cropSize, opt.cropSize),
                    (1, opt.cropSize, opt.cropSize)], 
                batch_size = 2, device=device.type)
        except RuntimeError as ex:
            print("ERROR in summary writer:", ex)

    #############################
    # NaN-hooks
    #############################
    CHECK_NAN_FORWARD = False

    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                print(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
                traceback.print_stack()
                raise RuntimeError("NaN!!")
    def add_nan_hooks(module):
        module.register_forward_hook(nan_hook)
    if CHECK_NAN_FORWARD:
        importanceModel.apply(add_nan_hooks)
        reconstructionModel.apply(add_nan_hooks)

    #############################
    # SHADING
    # for now, only used for testing
    #############################
    shading = ScreenSpaceShading(device)
    shading.fov(30)
    shading.ambient_light_color(np.array([0.1,0.1,0.1]))
    shading.diffuse_light_color(np.array([1.0, 1.0, 1.0]))
    shading.specular_light_color(np.array([0.2, 0.2, 0.2]))
    shading.specular_exponent(16)
    shading.light_direction(np.array([0.1,0.1,1.0]))
    shading.material_color(np.array([1.0, 0.3, 0.0]))
    shading.ambient_occlusion(1.0)
    shading.inverse_ao = False

    #############################
    # LOSSES
    #############################

    print('===> Building losses')
    if opt.mode == 'iso':
        criterion = losses.LossNetSparse(
            device,
            channels_out, 
            opt.losses,
            opt,
            has_flow = False)
        criterion.to(device)
    else: # dvr
        _criterion = losses.LossNet(
            device, 4, 4,
            None,
            opt.lossBorderPadding,
            opt.losses,
            opt)
        _criterion.to(device)
        criterion = lambda gt, pred, prev_pred_warped, no_temporal_loss, use_checkpoints, _crit=_criterion: \
            _crit(gt, pred, None, None, prev_pred_warped, use_checkpoints)
    print('Losses:', criterion)

    if opt.lossesTest is None:
        criterionTest = criterion
    else:
        if opt.mode == 'iso':
            criterionTest = losses.LossNetSparse(
                device,
                channels_out, 
                opt.losses + "," + opt.lossesTest,
                opt,
                has_flow = False)
            criterion.to(device)
        else: # dvr
            _criterion = losses.LossNet(
                device, 4, 4,
                None,
                opt.lossBorderPadding,
                opt.losses + "," + opt.lossesTest,
                opt)
            _criterion.to(device)
            criterionTest = lambda gt, pred, prev_pred_warped, no_temporal_loss, use_checkpoints, _crit=_criterion: \
                _crit(gt, pred, None, None, prev_pred_warped, use_checkpoints)

    #############################
    # OPTIMIZER
    #############################

    print('===> Create Optimizer ('+opt.optim+')')
    def createOptimizer(name, parameters, lr, weight_decay):
        if name=='Adam':
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif name=='RMSprop':
            return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        elif name=='Rprop':
            return optim.Rprop(parameters, lr=lr)
        elif name=='LBFGS':
            return optim.LBFGS(parameters, lr=lr)
        else:
            raise ValueError("Unknown optimizer "+name)
    if opt.importanceDontTrain:
        parameters = reconstructionModel.parameters()
    elif opt.reconDisable:
        parameters = importanceModel.parameters()
    else:
        parameters = itertools.chain(importanceModel.parameters(), reconstructionModel.parameters())
    optimizer = createOptimizer(opt.optim, parameters, 
                                lr=opt.lr, weight_decay=opt.weightDecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.lrStep, opt.lrGamma)

    #############################
    # PRETRAINED
    #############################
    if opt.pretrainedImportance is not None:
        checkpoint = torch.load(opt.pretrainedImportance)
        parameters = checkpoint['parameters']
        hasNormalization = 'normalize' in parameters
        #only load the state dict, not the whole model
        #this asserts that the model structure is the same
        model = importanceModel if hasNormalization else importanceModel._network
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].state_dict())
        elif 'importanceModel' in checkpoint:
            model.load_state_dict(checkpoint['importanceModel'].state_dict())
        else:
            raise ValueError("Can't find stored model with key 'model' or 'importanceModel' in the checkpoint for pretrained importance map models")
        print('Using pretrained model for the importance map model')

    if opt.pretrainedReconstruction is not None:
        checkpoint = torch.load(opt.pretrainedReconstruction)
        parameters = checkpoint['parameters']
        hasNormalization = 'normalize' in parameters
        #only load the state dict, not the whole model
        #this asserts that the model structure is the same
        model = reconstructionModel if hasNormalization else reconstructionModel._network
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].state_dict())
        elif 'reconstructionModel' in checkpoint:
            model.load_state_dict(checkpoint['reconstructionModel'].state_dict())
        else:
            raise ValueError("Can't find stored model with key 'model' or 'reconstructionModel' in the checkpoint for pretrained reconstruction models")
        print('Using pretrained model for the reconstruction model')

    #############################
    # OUTPUT DIRECTORIES or RESTORE
    #############################

    #Check for restoring
    startEpoch = 1
    if opt.restore is not None:
        if opt.runName is not None:
            restoreModelDir = os.path.join(opt.modeldir, opt.restore)
            if opt.restoreEpoch == -1:
                restoreEpoch = 0
                for fn in glob.glob(os.path.join(restoreModelDir, "model_epoch_*.pth")):
                    try:
                        e = int(fn[len(os.path.join(restoreModelDir, "model_epoch_")):-len(".pth")])
                        restoreEpoch = max(restoreEpoch, e)
                    except:
                        pass
            else:
                restoreEpoch = opt.restoreEpoch
            print("Restore training from run", opt.restore,"and epoch",restoreEpoch)
            modelInName = os.path.join(restoreModelDir, "model_epoch_{}.pth".format(restoreEpoch))
            checkpoint = torch.load(modelInName)
            importanceModel.load_state_dict(checkpoint['importanceModel'].state_dict())
            reconstructionModel.load_state_dict(checkpoint['reconstructionModel'].state_dict())
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
            startEpoch = restoreEpoch
        else:
            nextRunNumber = int(opt.restore)
            runName = 'run%05d'%nextRunNumber
            modeldir = os.path.join(opt.modeldir, runName)
            if opt.restoreEpoch == -1:
                restoreEpoch = 0
                while True:
                    modelInName = os.path.join(modeldir, "model_epoch_{}.pth".format(restoreEpoch+1))
                    if not os.path.exists(modelInName):
                        break;
                    restoreEpoch += 1
            else:
                restoreEpoch = opt.restoreEpoch

            print("Restore training from run", opt.restore,"and epoch",restoreEpoch)
            modelInName = os.path.join(modeldir, "model_epoch_{}.pth".format(restoreEpoch))
            checkpoint = torch.load(modelInName)
            importanceModel.load_state_dict(checkpoint['importanceModel'].state_dict())
            reconstructionModel.load_state_dict(checkpoint['reconstructionModel'].state_dict())
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
            startEpoch = restoreEpoch
            #paths
            print('Current run: %05d'%nextRunNumber)
            runName = 'run%05d'%nextRunNumber
            logdir = os.path.join(opt.logdir, runName)
            modeldir = os.path.join(opt.modeldir, runName)

    optStr = str(opt_dict);
    print(optStr)
    with open(os.path.join(modeldir, 'info.txt'), "w") as text_file:
        text_file.write(optStr+"\n")
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            text_file.write("git commit: " + git_commit + "\n")
        except:
            text_file.write("unable to get git commit\n")

    #tensorboard logger
    writer = SummaryWriter(logdir)
    writer.add_text('info', optStr, 0)
    # common hyperparameters
    important_hparams = [
        'pretrainedImportance', 'pretrainedReconstruction',
        'importanceNetUpscale', 'importancePostUpscale', 'importanceResidual',
        'sampleSharpness',
        'reconFilters', 'reconLayers',
        'importanceDisableTemporal', 'reconDisableTemporal',
        'reconResidual', 'reconInterpolateInput', 'reconPadding',
        'losses',
        'importanceMean', 'importanceMin']
    if opt.mode=='dvr':
        important_hparams += ['colorSpace']
    hparams_str = "  \n".join(["%s = %s"%(key, str(opt_dict[key])) for key in important_hparams])
    writer.add_text("hparam", hparams_str)
    #hparams_dict = {}
    #for hparam in important_hparams:
    #    hparams_dict['hparam/'+hparam] = str(opt_dict[hparam])
    #writer.add_hparams(hparams_dict, {})
    #for hparam in important_hparams:
    #    writer.add_text('hparam/'+hparam, str(opt_dict[hparam]))

    #############################
    # MAIN PART - evaluation function
    #############################

    def clampOutput(output):
        if opt.mode == 'iso':
            return torch.cat([
                torch.clamp(output[:,0:1,:,:], -1, +1), # mask
                ScreenSpaceShading.normalize(output[:,1:4,:,:], dim=1),
                torch.clamp(output[:,4:5,:,:], 0, +1), # depth
                torch.clamp(output[:,5:6,:,:], 0, +1) # ao
                ], dim=1)
        else: #dvr
            return torch.clamp(output, 0, 1) # rgba

    def evaluate(
        crop_low, pattern, crop_high,
        callback_fn, crit, *,
        hard_sample : bool,
        use_checkpoints : bool = False):
        """
        Evaluates the network and computes the losses.
        Since the evaluation is the same for training and testing,
        encapsulated in this function.

        Args:
         - crop_low, pattern, crop_high: the current batch from the data loader
         - callback_fn: function that takes as arguments:
            callback_fn(time, 
                importance_map, heatmapLoss, sample_mask, interpolated_samples,
                reconstruction, target, previous_input, 
                reconstructionLoss, reconstructionLossValues,
                importanceNormalizationStats)
         - crit: the loss function
         - hard_sample: True -> binary sampling
                        False -> relaxed sigmoid sampling (differentiable)
        Returns:
         the total loss of this batch
        """

        B, T, C, H, W = crop_high.shape

        pattern = pattern.unsqueeze(1)
        previous_importance = None
        previous_output = None
        loss = 0
        firstNet = True
        for j in range(T):
            # extract flow (always the last two channels of crop_high)
            flow = crop_high[:,j,C-2:,:,:]

            # select channels
            crop_high_out = crop_high[:, j, selectedChannelsOut, :, :]
            crop_high_in = crop_high[:, j, selectedChannelsIn, :, :]
            crop_low_in = crop_low[:, j, selectedChannelsIn, :, :]
            crop_target = crop_high_out

            # compute importance map
            if opt.importanceDontTrain:
                _importanceHadGradients = torch.is_grad_enabled()
                torch._C.set_grad_enabled(False)

            importance_input = crop_low_in
            if j==0 or opt.importanceDisableTemporal:
                previous_input = torch.zeros(
                    B,1,
                    importance_input.shape[2]*opt.importanceNetUpscale,
                    importance_input.shape[3]*opt.importanceNetUpscale, 
                    dtype=crop_high.dtype, device=crop_high.device)
            else:
                flow_low = F.interpolate(flow, scale_factor = 1/opt.importancePostUpscale)
                previous_input = models.VideoTools.warp_upscale(
                    previous_importance,
                    flow_low,
                    1, 
                    False)
            importance_input = torch.cat([
                importance_input,
                models.VideoTools.flatten_high(previous_input, opt.importanceNetUpscale)
                ], dim=1)
            importance_input = F.pad(importance_input, 
                                     (opt.importanceBorder, opt.importanceBorder, opt.importanceBorder, opt.importanceBorder),
                                     'constant', 0)
            # For some reasons, checkpointing does not work here
            #assert not torch.isnan(importance_input).any(), "NaN!!"
            if False: #use_checkpoints and importance_input.requires_grad and not opt.importanceDontTrain and not firstNet:
                importance_map = torch.utils.checkpoint.checkpoint(
                    importanceModel, importance_input) # the network call
            else:
                importance_map = importanceModel(importance_input) # the network call
            #assert not torch.isnan(importance_map).any(), "NaN!!"
            #if not opt.importanceDontTrain:
            #    firstNet = False

            #def visualizeImportance(map, name):
            #    import matplotlib.pyplot as plt
            #    plt.figure()
            #    if len(map.shape)==3:
            #        map = map[0]
            #    map = map.cpu().numpy()
            #    plt.imshow(map, vmin=0, vmax=1)
            #    plt.title(name)
            #visualizeImportance(importance_map[0], "Raw network output")

            importance_map = F.pad(importance_map, 
                                   [-opt.importanceBorder*opt.importanceNetUpscale]*4, 
                                   'constant', 0)
            #visualizeImportance(importance_map[0], "After padding")

            #previous_importance = importance_map
            heatmap_prior_value = importancePostprocess.importanceCenter()
            heatmapLoss = opt.lossHeatmapMean * ((heatmap_prior_value-torch.mean(importance_map))**2)
            loss = loss + heatmapLoss
            importance_map, importanceNormalizationStats = \
                importancePostprocess(importance_map) # normalization
            importance_map = torch.clamp(importance_map, max=1)
            importance_map = importance_map.unsqueeze(1)
            previous_importance = importance_map
            #print("importance map range: ({},{}), mean: {}".format(
            #    importance_map.min().item(), importance_map.max().item(), torch.mean(importance_map).item()))
            #visualizeImportance(importance_map[0], "After postprocess")


            if opt.importanceDontTrain:
                torch._C.set_grad_enabled(_importanceHadGradients)

            # create samples
            if hard_sample or opt.importanceDontTrain:
                sample_mask = (importance_map >= pattern).to(dtype=importance_map.dtype)
            else:
                sample_mask = torch.sigmoid(opt.sampleSharpness * (importance_map - pattern))
            #visualizeImportance(sample_mask[0], "Sample Mask")
            if opt.mode == 'iso':
                crop = crop_high_in
                crop = sample_mask * crop
                if opt.reconInterpolateInput: # interpolate crop
                    crop = importance.fractionalInpaint(crop, sample_mask[:,0,:,:])
                reconstruction_input = torch.cat((
                    crop, # mask, normal x, normal y, normal z, depth
                    torch.ones(B,1,H,W, dtype=crop_high.dtype, device=crop_high.device), # ao
                    sample_mask), # sample mask
                    dim = 1)
            else: #dvr
                crop = crop_high_in
                crop = sample_mask * crop
                if opt.reconInterpolateInput: # interpolate crop
                    crop = importance.fractionalInpaint(crop, sample_mask[:,0,:,:])
                reconstruction_input = torch.cat((
                    crop, # rgba
                    sample_mask), # sample mask
                    dim = 1)

            #plt.show()

            # warp previous output
            if j==0 or opt.reconDisableTemporal:
                previous_input = torch.zeros(B,channels_out,H,W, dtype=crop_high.dtype, device=crop_high.device)
            else:
                previous_input = models.VideoTools.warp_upscale(
                    previous_output,
                    flow,
                    1, False)

            # run reconstruction network
            if opt.reconDisable:
                reconstruction = reconstruction_input[:, :channels_out, :, :]
            else:
                reconstruction_input = torch.cat((reconstruction_input, previous_input), dim=1)
                if use_checkpoints and reconstruction_input.requires_grad and not firstNet:
                    reconstruction = torch.utils.checkpoint.checkpoint(
                        reconstructionModel, reconstruction_input, sample_mask)
                else:
                    reconstruction = reconstructionModel(reconstruction_input, sample_mask)
                firstNet = False

            # evaluate cost
            reconstructionLoss, reconstructionLossValues = crit(
                crop_target,
                reconstruction,
                previous_input,
                no_temporal_loss = (j==0 or opt.reconDisableTemporal),
                use_checkpoints = False)
            loss = loss + reconstructionLoss

            # send to callback for logging
            callback_fn(j, 
                        importance_map, heatmapLoss, sample_mask, crop,
                        reconstruction, crop_target, previous_input,
                        reconstructionLoss, reconstructionLossValues,
                        importanceNormalizationStats)

            # clamp outputs for the next frame
            previous_output = clampOutput(reconstruction)

            #loss.backward()
            #print("foo")

        # done with this batch
        return loss

    #############################
    # MAIN PART - training
    #############################

    def trainNormal(epoch):
        epoch_loss = 0
        avg_losses = defaultdict(float)
        num_minibatch = len(training_data_loader)
        pg = ProgressBar(num_minibatch, 'Training', length=50)

        stats_min = defaultdict(lambda: np.finfo(np.float32).max)
        stats_max = defaultdict(lambda: -np.finfo(np.float32).max)
        stats_avg = defaultdict(lambda: MeanVariance())

        importanceModel.train()
        reconstructionModel.train()
        for iteration, batch in enumerate(training_data_loader, 0):
            pg.print_progress_bar(iteration)
            crop_low, pattern, crop_high = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            num_frames = crop_low.shape[1]

            optimizer.zero_grad()

            def training_callback(
                    t, 
                    importance_map, heatmapLoss, sample_mask, interpolated_samples,
                    reconstruction, target, previous_input, 
                    reconstructionLoss, reconstructionLossValues,
                    importanceNormalizationStats):
                
                # losses
                for key, value in reconstructionLossValues.items():
                    avg_losses[str(key)] += value
                # save histogram of differences to the ground truth
                if opt.mode == 'iso':
                    residual = reconstruction[:,1:4,:,:] - interpolated_samples[:,1:4,:,:]
                else: # dvr
                    residual = reconstruction[:,0:3,:,:] - interpolated_samples[:,0:3,:,:]
                if iteration==0 and epoch>1:
                    residual = torch.clamp(residual, -0.2, +0.2)
                    if t==0:
                        writer.add_histogram('train/residual-values-t0', residual, epoch, bins='auto')
                    elif t==num_frames//2:
                        writer.add_histogram('train/residual-values-t%d'%t, residual, epoch, bins='auto')

                for stat, value in importanceNormalizationStats.items():
                    value = value.cpu().numpy().flat
                    B = len(value)
                    for b in range(B):
                        stats_min[stat] = min(stats_min[stat], value[b])
                        stats_max[stat] = max(stats_max[stat], value[b])
                        stats_avg[stat].append(float(value[b]))

            #with torch.autograd.detect_anomaly():
            if opt.reconDisableTemporal and opt.importanceDisableTemporal:
                # run each time slice independenty
                T = crop_low.shape[1]
                for t in range(T):
                    loss = evaluate(
                        crop_low[:, t:t+1, ...], pattern[:, t:t+1, ...], crop_high[:, t:t+1, ...], 
                        training_callback, 
                        criterion,
                        hard_sample=False,
                        use_checkpoints = opt.useCheckpointing)
                    epoch_loss += loss.item()
                    loss.backward()
            else:
                loss = evaluate(
                    crop_low, pattern, crop_high, 
                    training_callback, 
                    criterion,
                    hard_sample=False,
                    use_checkpoints = opt.useCheckpointing)
                epoch_loss += loss.item()
                loss.backward()
            optimizer.step()

        pg.print_progress_bar(num_minibatch)
        epoch_loss /= num_minibatch * num_frames
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))
        writer.add_scalar('train/total_loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        for stat in stats_avg.keys():
            writer.add_scalar('train/normalization-%s-min'%stat, stats_min[stat], epoch)
            writer.add_scalar('train/normalization-%s-max'%stat, stats_max[stat], epoch)
            writer.add_scalar('train/normalization-%s-avg'%stat, stats_avg[stat].mean(), epoch)
        # scalars
        for key in avg_losses.keys():
            avg_losses[key] /= num_minibatch * num_frames
        for key, value in avg_losses.items():
            writer.add_scalar('train/%s'%key, value, epoch)
        # todo: writer.add_hparams(...)
        writer.flush()
        scheduler.step()
   
    #############################
    # MAIN PART - testing
    #############################

    def test(epoch):
        def write_image(img, filename):
            out_img = img.cpu().detach().numpy()
            out_img *= 255.0
            out_img = out_img.clip(0, 255)
            out_img = np.uint8(out_img)
            writer.add_image(filename, out_img, epoch)

            #plt.figure()
            #if out_img.shape[0]==1:
            #    out_img = out_img[0]
            #if len(out_img.shape)==3:
            #    out_img = out_img.transpose((1,2,0))
            #plt.imshow(out_img)
            #plt.title(filename)

        avg_psnr = 0
        avg_losses = defaultdict(float)
        heatmap_min = 1e10
        heatmap_max = -1e10
        heatmap_avg = heatmap_count = 0
        with torch.no_grad():
            num_minibatch = len(testing_data_loader)
            pg = ProgressBar(num_minibatch, 'Testing', length=50)
            importanceModel.eval()
            reconstructionModel.eval()
            epoch_loss = 0
            for iteration, batch in enumerate(testing_data_loader, 0):
                pg.print_progress_bar(iteration)
                crop_low, pattern, crop_high = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                num_frames = crop_low.shape[1]

                def testing_callback(
                        t, 
                        importance_map, heatmapLoss, sample_mask, interpolated_samples,
                        reconstruction, target, previous_input, 
                        reconstructionLoss, reconstructionLossValues,
                        importanceNormalizationStats):
                    # heatmap / importance map
                    nonlocal heatmap_min, heatmap_max, heatmap_avg, heatmap_count, avg_losses
                    heatmap_min = min(heatmap_min, torch.min(importance_map).item())
                    heatmap_max = max(heatmap_max, torch.max(importance_map).item())
                    heatmap_avg += torch.mean(importance_map).item()
                    heatmap_count += 1
                    # other losses
                    avg_losses['total_loss'] += heatmapLoss.item() + reconstructionLoss.item()
                    avg_losses['heatmap_loss'] += heatmapLoss.item()
                    if opt.mode == 'iso':
                        psnr = 10 * log10(1 / max(1e-10, reconstructionLossValues[('mse','color')]))
                    else:
                        psnr = 10 * log10(1 / max(1e-10, reconstructionLossValues['mse']))
                    avg_losses['psnr'] += psnr
                    for key, value in reconstructionLossValues.items():
                        avg_losses[str(key)] += value
                    # save histogram of differences to the input (residual)
                    if opt.mode == 'iso':
                        residual = reconstruction[:,1:4,:,:] - interpolated_samples[:,1:4,:,:]
                    else: # dvr
                        residual = reconstruction[:,0:3,:,:] - interpolated_samples[:,0:3,:,:]
                    if iteration==0 and epoch>1:
                        residual = torch.clamp(residual, -0.2, +0.2)
                        if t==0:
                            writer.add_histogram('test/residual-values-t0', residual, epoch, bins='auto')
                        elif t==num_frames//2:
                            writer.add_histogram('test/residual-values-t%d'%t, residual, epoch, bins='auto')
                    # save histogram of sampling
                    #TODO
                    # save images
                    B = target.shape[0]
                    imagesToSave = opt.numVisImages - iteration*opt.testBatchSize
                    if imagesToSave>0 and (epoch<opt.visImagesFreq or epoch%opt.visImagesFreq==0):
                        reconstruction = clampOutput(reconstruction)
                        # for each image in the batch
                        for b in range(min(B, imagesToSave)):
                            imgID = b + iteration * B
                            # importance map + sample mask
                            samplingImg = torch.cat([importance_map[b,...], sample_mask[b,...], target[b,0:1,:,:]], dim=2)
                            write_image(samplingImg, 'image%03d/sampling/frame%03d' % (imgID, t))
                            
                            if opt.mode == 'iso':
                                # reconstructions
                                if opt.reconDisableTemporal:
                                    # images, two in a row: current prediction, ground truth
                                    maskPredGT = torch.cat([reconstruction[b,0:1,:,:], target[b,0:1,:,:]], dim=2)*0.5+0.5
                                    write_image(maskPredGT, 'image%03d/mask/frame%03d' % (imgID, t))
                                    normalPredGT = torch.cat([reconstruction[b,1:4,:,:], target[b,1:4,:,:]], dim=2)*0.5+0.5
                                    write_image(normalPredGT, 'image%03d/normal/frame%03d' % (imgID, t))
                                    depthPredGT = torch.cat([reconstruction[b,4:5,:,:], target[b,4:5,:,:]], dim=2)
                                    write_image(depthPredGT, 'image%03d/depth/frame%03d' % (imgID, t))
                                    aoPredGT = torch.cat([reconstruction[b,5:6,:,:], target[b,5:6,:,:]], dim=2)
                                    write_image(aoPredGT, 'image%03d/ao/frame%03d' % (imgID, t))
                                else:
                                    # images, three in a row: previous-warped, current prediction, ground truth
                                    maskPredGT = torch.cat([previous_input[b,0:1,:,:], reconstruction[b,0:1,:,:], target[b,0:1,:,:]], dim=2)*0.5+0.5
                                    write_image(maskPredGT, 'image%03d/mask/frame%03d' % (imgID, t))
                                    normalPredGT = torch.cat([previous_input[b,1:4,:,:], reconstruction[b,1:4,:,:], target[b,1:4,:,:]], dim=2)*0.5+0.5
                                    write_image(normalPredGT, 'image%03d/normal/frame%03d' % (imgID, t))
                                    depthPredGT = torch.cat([previous_input[b,4:5,:,:], reconstruction[b,4:5,:,:], target[b,4:5,:,:]], dim=2)
                                    write_image(depthPredGT, 'image%03d/depth/frame%03d' % (imgID, t))
                                    aoPredGT = torch.cat([previous_input[b,5:6,:,:], reconstruction[b,5:6,:,:], target[b,5:6,:,:]], dim=2)
                                    write_image(aoPredGT, 'image%03d/ao/frame%03d' % (imgID, t))
                                if opt.reconInterpolateInput and opt.reconResidual:
                                    # draw interpolated input, reconstruction, residual
                                    normalInputResidual = torch.cat([
                                        interpolated_samples[b,1:4,:,:], 
                                        reconstruction[b,1:4,:,:], 
                                        (reconstruction[b,1:4,:,:] - interpolated_samples[b,1:4,:,:]) + 0.5
                                        ], dim=2)*0.5+0.5
                                    write_image(normalInputResidual, 'image%03d/interpolation/frame%03d' % (imgID, t))
                                elif opt.reconInterpolateInput:
                                    # draw interpolated input, reconstruction (no residual)
                                    normalInputResidual = torch.cat([
                                        interpolated_samples[b,1:4,:,:], 
                                        reconstruction[b,1:4,:,:]
                                        ], dim=2)*0.5+0.5
                                    write_image(normalInputResidual, 'image%03d/interpolation/frame%03d' % (imgID, t))
                            else: #dvr
                                if opt.reconDisableTemporal:
                                    # images, two in a row: current prediction, ground truth
                                    alphaPredGT = torch.cat([reconstruction[b,3:4,:,:], target[b,3:4,:,:]], dim=2)
                                    write_image(alphaPredGT, 'image%03d/alpha/frame%03d' % (imgID, t))
                                    colorPredGT = torch.cat([reconstruction[b,0:3,:,:], target[b,0:3,:,:]], dim=2)
                                    write_image(colorPredGT, 'image%03d/color/frame%03d' % (imgID, t))
                                else:
                                    # images, three in a row: previous-warped, current prediction, ground truth
                                    alphaPredGT = torch.cat([previous_input[b,3:4,:,:], reconstruction[b,3:4,:,:], target[b,3:4,:,:]], dim=2)
                                    write_image(alphaPredGT, 'image%03d/alpha/frame%03d' % (imgID, t))
                                    colorPredGT = torch.cat([previous_input[b,0:3,:,:], reconstruction[b,0:3,:,:], target[b,0:3,:,:]], dim=2)
                                    write_image(colorPredGT, 'image%03d/color/frame%03d' % (imgID, t))
                                if opt.reconInterpolateInput and opt.reconResidual:
                                    # draw interpolated input, reconstruction, residual
                                    colorInputResidual = torch.cat([
                                        interpolated_samples[b,0:3,:,:], 
                                        reconstruction[b,0:3,:,:], 
                                        (reconstruction[b,0:3,:,:] - interpolated_samples[b,0:3,:,:]) + 0.5
                                        ], dim=2)
                                    write_image(colorInputResidual, 'image%03d/interpolation/frame%03d' % (imgID, t))
                                elif opt.reconInterpolateInput:
                                    # draw interpolated input, reconstruction (no residual)
                                    colorInputResidual = torch.cat([
                                        interpolated_samples[b,0:3,:,:], 
                                        reconstruction[b,0:3,:,:]
                                        ], dim=2)
                                    write_image(colorInputResidual, 'image%03d/interpolation/frame%03d' % (imgID, t))

                loss = evaluate(
                    crop_low, pattern, crop_high, 
                    testing_callback, 
                    criterionTest,
                    hard_sample=True,
                    use_checkpoints=False)
                epoch_loss += loss.item()
            pg.print_progress_bar(num_minibatch)

            # scalars
            for key in avg_losses.keys():
                avg_losses[key] /= num_minibatch * num_frames
            print("===> Avg. PSNR: {:.4f} dB".format(avg_losses['psnr']))
            print("  losses:",avg_losses)
            for key, value in avg_losses.items():
                writer.add_scalar('test/%s'%key, value, epoch)
            print("  heatmap: min=%f, max=%f, avg=%f"%(heatmap_min, heatmap_max, heatmap_avg/heatmap_count))
            writer.add_scalar('test/heatmap_min', heatmap_min, epoch)
            writer.add_scalar('test/heatmap_max', heatmap_max, epoch)
            writer.add_scalar('test/heatmap_avg', heatmap_avg/heatmap_count, epoch)
            # todo: writer.add_hparams(...)
            writer.flush()

    def checkpoint(epoch):
        model_out_path = os.path.join(modeldir, "model_epoch_{}.pth".format(epoch))
        state = {
            'epoch': epoch + 1, 
            'importanceModel': importanceModel,
            'reconstructionModel': reconstructionModel,
            'parameters':opt_dict,
            'optimizer':optimizer, 
            'scheduler':scheduler,
            'normalizationInfo':normalizationInfo.getParameters()}
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    if not os.path.exists(opt.modeldir):
        os.mkdir(opt.modeldir)
    if not os.path.exists(opt.logdir):
        os.mkdir(opt.logdir)

    print('===> Start Training')

    try:
        test(0)
        for epoch in range(startEpoch, opt.nEpochs + 1):
            trainNormal(epoch)
            test(epoch)
            if epoch%10==0:
                checkpoint(epoch)
    except KeyboardInterrupt:
        print("INTERRUPT, stop training")
        writer.close()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        with open(os.path.join(logdir, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
            f.write("\n")
        writer.close()
        raise