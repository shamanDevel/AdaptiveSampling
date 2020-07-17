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
from typing import List, Callable

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

    parser.add_argument('--mode', type=str, choices=['iso', 'dvr'], default='dvr', help="""
        Current, only DVR is supported for stepsize training.""")

    #Dataset
    parser_group = parser.add_argument_group("Dataset")
    parser_group.add_argument('--datasetTarget', type=str, required=True,
                        help="Path the the HDF5 file with the target outputs divided by steps. Format of the entries: step%d")
    parser_group.add_argument('--datasetInput', type=str, required=True, help="""
        Path the the HDF5 file with the dataset of the input image, given in the dset called 'gt'.
        The difference in resolution between datasetTarget and datasetInput specifies the upscaling factor of the importance network""")
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

    #Restore
    parser_group = parser.add_argument_group("Restore")
    parser_group.add_argument('--restore', type=int, default=-1, help="Restore training from the specified run index")
    parser_group.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
    parser_group.add_argument('--pretrainedImportance', type=str, default=None, 
                              help="Path to a pretrained importance-map network")
    parser_group.add_argument('--pretrainedReconstruction', type=str, default=None, 
                              help="Path to a pretrained importance-map network")

    #Importance Model parameters
    parser_group = parser.add_argument_group("Importance-Model")
    parser_group.add_argument('--importanceModel', type=str, default="EnhanceNet", choices=['EnhanceNet'], help="""
    The model for importance map generation, support is only 'EnhanceNet' so far.""")
    parser_group.add_argument('--importanceLayers', type=int, default=5, help="Number of layers in the importance map network")
    parser_group.add_argument('--importanceUseBN', action='store_true', help='Enable batch normalization in the generator and discriminator')
    parser_group.add_argument('--importanceBorder', type=int, default=8, help='Zero border around the network')
    parser_group.add_argument('--importanceOutputLayer', type=str, default='softplus', choices=['softplus'], help="""
        Network output layer, either 'none', 'softplus' or 'sigmoid'.
        Only 'softplus' is supported.""")
    parser_group.add_argument('--importanceMin', type=float, default=0.01,
                              help="the minimal importance value, i.e. the maximal spacing of samples")
    parser_group.add_argument('--importanceMean', type=float, default=0.2,
                              help="the mean importance value, i.e. the average number of samples")
    parser_group.add_argument('--importanceDisableTemporal', action='store_true', help='Disables temporal consistency')
    parser_group.add_argument('--importanceDontTrain', action='store_true',
                              help="Disables training of the importance sampler")

    # Reconstruction Model
    parser_group = parser.add_argument_group("Reconstruction-Model")
    parser_group.add_argument('--reconModel', type=str, default='EnhanceNet', choices=['UNet', 'EnhanceNet', 'Baseline'], help="""
        The network architecture, supported are 'UNet' and 'EnhanceNet', 'Baseline'.
        'Baseline' simply passes the input to the output without changing anything.""")
    parser_group.add_argument('--reconLayers', type=int, default=10, help="The depth of the network")
    parser_group.add_argument('--reconFilters', type=int, default=64, help=""" 
            UNet: an int, the number of filters in the first layer of the UNet is 2**this_value.
            EnhanceNet: the number of features per layer.""")
    parser_group.add_argument('--reconPadding', type=str, default="zero", choices=['off','zero','partial'],
                                help="UNet: The padding mode for the UNet. 'partial' can only be used together with --importanceDontTrain")
    parser_group.add_argument('--reconUseBN', action='store_true', help="UNet: Use batch normalization in the network")
    parser_group.add_argument('--reconResidual', action='store_true', help="""
        Use residual connections from input to output.""")
    parser_group.add_argument('--reconUpMode', type=str, default='upsample', choices=['upconv', 'upsample'],
                                help="UNet: The upsample mode")
    parser_group.add_argument('--reconDisableTemporal', action='store_true', help='Disables temporal consistency')
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
    opt_dict['type'] = 'stepsize1'
    """
    adaptive1: importance model has no temporal consistency
    adaptive2: importance model with temporal consistency
    """

    #########################
    # BASIC CONFIGURATION
    #########################

    # load renderer
    requires_renderer = False # not needed at the moment
    if requires_renderer:
        print("Renderer required")
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

    if opt.reconPadding == 'partial' and not opt.importanceDontTrain:
        raise ValueError("reconstruction padding mode 'partial' can only be used with --importanceDontTrain")

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
            else:
                print("ERROR: specified run directory already exists!")
                exit(-1)
        else:
            os.makedirs(logdir)
            os.makedirs(modeldir)
    elif opt.restore == -1:
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
        print("Iso-Surfaces are not supported anymore")
        selectedChannelsInList = (opt.inputChannels or "normal,depth").split(',')
        selectedChannelsIn = [('mask', 0)]
        if "normal" in selectedChannelsInList:
            selectedChannelsIn += [('normalX', 1), ('normalY', 2), ('normalZ', 3)]
            residualChannels = [1,2,3]
        if "depth" in selectedChannelsInList:
            selectedChannelsIn += [('depth', 4)]
        channels_low = len(selectedChannelsIn) # mask, normal xyz, depth
        channels_sampling = channels_low
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
        channels_sampling = channels_low
        channels_out = 4 # red green blue alpha
        selectedChannelsOut = list(range(4))
    else:
        raise ValueError("unknown mode "+opt.mode)
    print("selected input channels:", selectedChannelsIn)
    selectedChannelsIn = [t[1] for t in selectedChannelsIn]


    #########################
    # DATASETS
    #########################

    print('===> Loading datasets')

    # find crops in the images
    image_crops = dataset.datasetUtils.getCropsForDataset(
        opt.datasetTarget, 'step0',
        opt.samples, opt.cropSize,
        1, opt.cropFillRate, 
        0 if opt.mode=='iso' else 3) # mask: iso -> 0 (mask, normal, ...), dvr -> 3 (rgb, alpha)

    # create dataset
    l = int(opt.samples * opt.testFraction)
    train_set = dataset.stepsizeDatasetLoader.StepsizeDataset(
        crop_size = opt.cropSize,
        dataset_target = opt.datasetTarget,
        dataset_input = opt.datasetInput,
        data_crops = image_crops[l:,:])
    test_set = dataset.stepsizeDatasetLoader.StepsizeDataset(
        crop_size = opt.cropSize,
        dataset_target = opt.datasetTarget,
        dataset_input = opt.datasetInput,
        data_crops = image_crops[:l,:])
    dataset_min_stepsize = train_set.min_stepsize()
    dataset_max_stepsize = train_set.max_stepsize()
    dataset_upscale_factor = train_set.upscale_factor()
    opt_dict['importanceUpscale'] = dataset_upscale_factor

    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.trainBatchSize, shuffle=True, num_workers=opt.numWorkers)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.numWorkers)

    channels_low_with_previous = channels_low + dataset_upscale_factor**2
    channels_sampling_with_previous = channels_sampling + channels_out

    #############################
    # IMPORTANCE-MAP MODEL
    #############################
    print('===> Building importance-map model')
    importanceModel = importance.NetworkImportanceMap(
        dataset_upscale_factor, 
        channels_low_with_previous,
        model = opt.importanceModel,
        use_bn = opt.importanceUseBN,
        border_padding = opt.importanceBorder,
        output_layer = opt.importanceOutputLayer,
        num_layers = opt.importanceLayers,
        residual = 'off',
        residual_channels = residualChannels)
    importancePostprocess = importance.PostProcess(
            opt.importanceMin, opt.importanceMean, 
            1, # post-upscale
            0, # padding
            'basic') # normalization mode

    importanceModel.to(device)
    print('Importance-Map Model:')
    print(importanceModel)
    if not no_summary:
        try:
            summary(importanceModel, 
                input_size=(channels_low_with_previous, 
                            (opt.cropSize+2*opt.importanceBorder)//dataset_upscale_factor, 
                            (opt.cropSize+2*opt.importanceBorder)//dataset_upscale_factor), 
                batch_size = 2, device=device.type)
        except RuntimeError as ex:
            print("ERROR in summary writer:", ex)

    #############################
    # RECONSTRUCTION MODEL
    #############################
    print('===> Building reconstruction model')
    class EnhanceNetWrapper(nn.Module):
        def __init__(self, orig):
            super().__init__();
            self._orig = orig
        def forward(self, input):
            return self._orig(input)[0]
    if opt.reconModel == "UNet":
        reconstructionModel = models.UNet(
            in_channels = channels_sampling_with_previous, 
            out_channels = channels_out,
            depth = opt.reconLayers, 
            wf = opt.reconFilters, 
            padding = opt.reconPadding,
            batch_norm = opt.reconUseBN, 
            residual = opt.reconResidual,
            hard_input = False, 
            up_mode = opt.reconUpMode,
            return_masks = False)
    elif opt.reconModel == "EnhanceNet":
        ReconOpt = namedtuple("ReconOpt", ["useBN", "num_layers", "num_channels"])
        reconOpt = ReconOpt(
            useBN = opt.reconUseBN,
            num_layers = opt.reconLayers,
            num_channels = opt.reconFilters)
        _reconstructionModel = models.EnhanceNet(
            upscale_factor = 1,
            input_channels = channels_sampling_with_previous,
            channel_mask = list(range(channels_out)),
            output_channels = channels_out,
            opt=reconOpt)
        reconstructionModel = EnhanceNetWrapper(_reconstructionModel)
    elif opt.reconModel == "Baseline":
        class BaselineReconModel(nn.Module):
            def __init__(self, selectedChannelsOut):
                super().__init__();
                self._selectedChannelsOut = selectedChannelsOut
            def forward(self, input):
                return input[:, self._selectedChannelsOut, :, :]
        reconstructionModel = BaselineReconModel(selectedChannelsOut)
    else:
        raise ValueError("Unsupported reconstruction model "+opt.reconModel)

    reconstructionModel.to(device)
    print('Reconstruction Model:')
    print(reconstructionModel)
    if not no_summary:
        try:
            summary(reconstructionModel, 
                input_size=
                    (channels_sampling_with_previous, opt.cropSize, opt.cropSize), 
                batch_size = 2, device=device.type)
        except (RuntimeError, AttributeError) as ex:
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
    if opt.restore != -1:
        nextRunNumber = opt.restore
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
        'importanceUpscale',
        'reconFilters', 'reconLayers',
        'importanceDisableTemporal', 'reconDisableTemporal',
        'reconResidual', 'reconPadding',
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
        batch_input : torch.Tensor, 
        batch_targets : List[torch.Tensor],
        callback_fn, 
        crit : Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool], torch.Tensor],
        *,
        use_checkpoints : bool = False):
        """
        Evaluates the network and computes the losses.
        Since the evaluation is the same for training and testing,
        encapsulated in this function.

        Args:
         - batch_input, batch_targets: the current batch from the data loader
         - callback_fn: function that takes as arguments:
            callback_fn(time, 
                        importance_input, importance_map, heatmapLoss,
                        reconstruction_input,
                        reconstruction, previous_input,
                        crop_target_out,
                        reconstructionLoss, reconstructionLossValues,
                        importanceNormalizationStats)
         - crit: the loss function: (ground truth, prediction, previous-prediction, no-temporal-loss, use-checkponts)
        Returns:
         the total loss of this batch
        """

        B, T, C, H, W = batch_targets[0].shape
        C = len(selectedChannelsIn)

        previous_importance = None
        previous_output = None
        loss = 0
        firstNet = True
        for j in range(T):
            # extract flow (always the last two channels of crop_high)
            flow = batch_targets[0][:,j,-2:,:,:]

            # select channels
            crop_low_in = batch_input[:, j, selectedChannelsIn, :, :]
            crops_sampling_in = [t[:, j, selectedChannelsIn, :, :] for t in batch_targets]
            crop_target_out = batch_targets[0][:, j, selectedChannelsOut, :, :]

            ######################################################################
            # IMPORTANCE MAP
            ######################################################################

            if opt.importanceDontTrain:
                _importanceHadGradients = torch.is_grad_enabled()
                torch._C.set_grad_enabled(False)

            importance_input = crop_low_in
            if j==0 or opt.importanceDisableTemporal:
                previous_input = torch.zeros(
                    B,1,
                    importance_input.shape[2]*dataset_upscale_factor,
                    importance_input.shape[3]*dataset_upscale_factor, 
                    dtype=crop_low_in.dtype, device=crop_low_in.device)
            else:
                previous_input = models.VideoTools.warp_upscale(
                    previous_importance,
                    flow,
                    dataset_upscale_factor, 
                    False)
            importance_input = torch.cat([
                importance_input,
                models.VideoTools.flatten_high(previous_input, dataset_upscale_factor)
                ], dim=1)
            # For some reasons, checkpointing does not work here
            #assert not torch.isnan(importance_input).any(), "NaN!!"
            importance_input = F.pad(importance_input, 
                                     (opt.importanceBorder, opt.importanceBorder, opt.importanceBorder, opt.importanceBorder),
                                     'constant', 0)
            if False: #use_checkpoints and importance_input.requires_grad and not opt.importanceDontTrain and not firstNet:
                importance_map = torch.utils.checkpoint.checkpoint(
                    importanceModel, importance_input) # the network call
            else:
                importance_map = importanceModel(importance_input) # the network call
            importance_map = F.pad(importance_map, 
                                   [-opt.importanceBorder*dataset_upscale_factor]*4, 
                                   'constant', 0)
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

            #previous_importance = importance_map
            heatmap_prior_value = importancePostprocess.importanceCenter()
            heatmapLoss = opt.lossHeatmapMean * ((heatmap_prior_value-torch.mean(importance_map))**2)
            loss = loss + heatmapLoss
            importance_map, importanceNormalizationStats = \
                importancePostprocess(importance_map) # normalization
            previous_importance = importance_map.unsqueeze(1)
            #print("importance map range: ({},{}), mean: {}".format(
            #    importance_map.min().item(), importance_map.max().item(), torch.mean(importance_map).item()))
            #visualizeImportance(importance_map[0], "After postprocess")

            if opt.importanceDontTrain:
                torch._C.set_grad_enabled(_importanceHadGradients)

            ######################################################################
            # STEPSIZE SAMPLING
            ######################################################################

            # Inputs:
            # importance map in (B, H, W)
            # crops_sampling_in in [(B, C, H, W)*S]
            # where S is the number of available step sizes (dataset_max_stepsize-dataset_min_stepsize+1)

            # the importance map gives the number of samples allowed at that position
            # Hence, the step size is
            step_size = 1.0 / importance_map
            # The dataloader provides images for stepsizes being a power-of-two step_size=2^n with i in Z
            step_index = torch.log2(step_size)
            # clamp it to the minimal and maximal indices that the dataloader provides
            step_index = torch.clamp(step_index, min=dataset_min_stepsize, max=dataset_max_stepsize) - dataset_min_stepsize
            # get fractional position between floor and ceiling, this is the differentiable part
            step_index_low = torch.floor(step_index) # shape (B, H, W)
            step_index_high = torch.ceil(step_index)
            step_index_fractional = step_index - step_index_low
            step_index_low = step_index_low.to(dtype=torch.long) # shape (B, H, W)
            step_index_high = step_index_high.to(dtype=torch.long)
            # assemble the target crops in a tensor so that I can index them properly
            crops_sampling_in_tensor = torch.stack(crops_sampling_in, dim=0) # shape (S, B, C, H, W)
            # flatten batch, width, height so that I can call index_select
            flat_input = crops_sampling_in_tensor.transpose(1, 2).flatten(2) # shape (S, C, B*H*W)
            step_index_low2 = torch.stack([step_index_low.flatten()]*C, dim=0).unsqueeze(0) # shape (1, C, B*H*W)
            step_index_high2 = torch.stack([step_index_high.flatten()]*C, dim=0).unsqueeze(0) # shape (1, C, B*H*W)
            step_index_fractional2 = step_index_fractional.flatten().unsqueeze(0) # shape (1, B*H*W)
            # Now fetch the samples and interpolate
            samples_low = torch.gather(flat_input, 0, step_index_low2)[0] # shape (C, B*H*W)
            samples_high = torch.gather(flat_input, 0, step_index_high2)[0]
            samples_interpolated = samples_low + step_index_fractional2 * (samples_high - samples_low)
            # unflatten and transpose
            reconstruction_input = samples_interpolated.reshape(C, B, H, W).transpose(0, 1) # shape (B, C, H, W)

            ######################################################################
            # RECONSTRUCTION
            ######################################################################

            # warp previous output
            if j==0 or opt.reconDisableTemporal:
                previous_input = torch.zeros(B,channels_out,H,W, dtype=crop_low_in.dtype, device=crop_low_in.device)
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
                        reconstructionModel, reconstruction_input)
                else:
                    reconstruction = reconstructionModel(reconstruction_input)
                firstNet = False

            # evaluate cost
            reconstructionLoss, reconstructionLossValues = crit(
                crop_target_out,
                reconstruction,
                previous_input,
                no_temporal_loss = (j==0 or opt.reconDisableTemporal),
                use_checkpoints = False)
            loss = loss + reconstructionLoss

            # send to callback for logging
            callback_fn(j, 
                        importance_input, importance_map, heatmapLoss,
                        reconstruction_input,
                        reconstruction, previous_input,
                        crop_target_out,
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
        num_minibatch = len(training_data_loader)
        pg = ProgressBar(num_minibatch, 'Training', length=50)

        stats_min = defaultdict(lambda: np.finfo(np.float32).max)
        stats_max = defaultdict(lambda: -np.finfo(np.float32).max)
        stats_avg = defaultdict(lambda: MeanVariance())

        importanceModel.train()
        reconstructionModel.train()
        for iteration, batch in enumerate(training_data_loader, 0):
            pg.print_progress_bar(iteration)
            crop_input, crops_target = batch[0].to(device), [t.to(device) for t in batch[1]]
            num_frames = crop_input.shape[1]

            optimizer.zero_grad()

            def training_callback(
                    time, 
                    importance_input, importance_map, heatmapLoss,
                    reconstruction_input,
                    reconstruction, previous_input,
                    crop_target_out,
                    reconstructionLoss, reconstructionLossValues,
                    importanceNormalizationStats):
                
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
                        crop_input[:, t:t+1, ...], crops_target[:, t:t+1, ...],
                        training_callback, 
                        criterion,
                        hard_sample=False,
                        use_checkpoints = opt.useCheckpointing)
                    epoch_loss += loss.item()
                    loss.backward()
            else:
                loss = evaluate(
                    crop_input, crops_target,
                    training_callback, 
                    criterion,
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
                crop_input, crops_target = batch[0].to(device), [t.to(device) for t in batch[1]]
                num_frames = crop_input.shape[1]

                def testing_callback(
                        t, 
                        importance_input, importance_map, heatmapLoss,
                        reconstruction_input,
                        reconstruction, previous_input,
                        target,
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
                    # save images
                    B = target.shape[0]
                    imagesToSave = opt.numVisImages - iteration*opt.testBatchSize
                    if imagesToSave>0 and (epoch<opt.visImagesFreq or epoch%opt.visImagesFreq==0):
                        reconstruction = clampOutput(reconstruction)
                        # for each image in the batch
                        for b in range(min(B, imagesToSave)):
                            imgID = b + iteration * B
                            # importance map + sample mask
                            #importance_input_scaled = F.interpolate(importance_input[b:b+1,0:3,...], scale_factor=(dataset_upscale_factor,dataset_upscale_factor), mode='nearest')
                            samplingImg = torch.cat([torch.stack([importance_map[b,...]*0.2]*3, dim=0), reconstruction_input[b,0:3,:,:], target[b,0:3,:,:]], dim=2)
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
                                if opt.reconResidual:
                                    # draw interpolated input, reconstruction, residual
                                    normalInputResidual = torch.cat([
                                        reconstruction_input[b,1:4,:,:],
                                        reconstruction[b,1:4,:,:], 
                                        (reconstruction[b,1:4,:,:] - reconstruction_input[b,1:4,:,:]) + 0.5
                                        ], dim=2)*0.5+0.5
                                    write_image(normalInputResidual, 'image%03d/residual/frame%03d' % (imgID, t))
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
                                if opt.reconResidual:
                                    # draw interpolated input, reconstruction, residual
                                    colorInputResidual = torch.cat([
                                        reconstruction_input[b,0:3,:,:], 
                                        reconstruction[b,0:3,:,:], 
                                        (reconstruction[b,0:3,:,:] - reconstruction_input[b,0:3,:,:])+0.5
                                        ], dim=2)
                                    write_image(colorInputResidual, 'image%03d/residual/frame%03d' % (imgID, t))

                loss = evaluate(
                    crop_input, crops_target, 
                    testing_callback, 
                    criterionTest,
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
            'scheduler':scheduler}
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