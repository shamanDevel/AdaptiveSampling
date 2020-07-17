from __future__ import print_function
import argparse
import math
from math import log10
import os
import os.path
from collections import defaultdict
import itertools
import sys
import traceback
import subprocess
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

no_summary = False
try:
    from torchsummary import summary
except ModuleNotFoundError:
    no_summary = True
    print("No summary writer found")

from console_progressbar import ProgressBar

#from data import get_training_set, get_test_set
import dataset
import models
import losses
from utils import ScreenSpaceShading, initialImage, getColorSpaceConversions

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Superresolution for Isosurface Raytracing')

    parser.add_argument('--mode', type=str, choices=['iso', 'dvr'], required=True, help="""
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

    parser_group = parser.add_argument_group("Dataset")
    parser_group.add_argument('--targetDataset', type=str,
                        help="Path the the HDF5 file with the target dataset")
    parser_group.add_argument('--inputDataset', type=str,
                        help="Path the the HDF5 file with the input dataset")
    parser_group.add_argument('--upscaleFactor', type=int, default=4, help="super resolution upscale factor")
    parser_group.add_argument('--cropSize', type=int, default=128, help="The size of the high-resolution crop")
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
        Specifies the channels which should be normalized before sending to the network
        as a comma-separated list. Default: no normalization / empty list.""")

    parser_group = parser.add_argument_group("Restore")
    parser_group.add_argument('--restore', type=int, default=-1, help="Restore training from the specified run index")
    parser_group.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
    parser_group.add_argument('--pretrained', type=str, default=None, help="Path to a pretrained generator")
    parser_group.add_argument('--pretrainedDiscr', type=str, default=None, help="Path to a pretrained discriminator")

    #Model parameters
    parser_group = parser.add_argument_group("Model")
    parser_group.add_argument('--model', type=str, default="EnhanceNet", help="""
        The superresolution model.
        Supported nets: 'SubpixelNet', 'EnhanceNet', 'TecoGAN', 'RCAN'
        """)
    parser_group.add_argument('--upsample', type=str, default='bilinear', help='Upsampling for EnhanceNet: nearest, bilinear, bicubic, or pixelShuffle')
    parser_group.add_argument('--reconType', type=str, default='residual', help='Block type for EnhanceNet: residual or direct')
    parser_group.add_argument('--useBN', action='store_true', help='Enable batch normalization in the generator and discriminator')
    parser_group.add_argument('--useSN', action='store_true', help='Enable spectral normalization in the generator and discriminator')
    parser_group.add_argument('--numResidualLayers', type=int, default=10, help='Number of residual layers in the generator')
    parser_group.add_argument('--numFilters', type=int, default=32, help='Number of filter channels in the generator')
    parser_group.add_argument('--disableTemporal', action='store_true', help='Disables temporal consistency')
    parser_group.add_argument('--initialImage', type=str, default='zero', help="""
    Specifies what should be used as the previous high res frame for the first frame of the sequence,
    when no previous image is available from the previous predition.
    Available options:
     - zero: fill everything with zeros
     - unshaded: Special defaults for unshaded mode: mask=-1, normal=[0,0,1], depth=0.5, ao=1
     - input: Upscale the current input image and use that as the previous high res image
       Remaining channels are filled with defaults
    Default: 'input'
    """)

    #Loss parameters
    parser_group = parser.add_argument_group("Loss")
    parser_group.add_argument('--losses', type=str, required=True, help="""
    Comma-separated list of loss functions: 
    l1,l2,ssim,lpips,perceptual,texture,adv. 
    Optinally, the weighting factor can be specified with a colon.
    Example: "--losses perceptual:0.1,texture:1e2,adv:10"
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
    parser_group.add_argument('--discriminator', type=str, default='enhanceNetLarge', help="""
    Network architecture for the discriminator.
    Possible values: enhanceNetSmall, enhanceNetLarge, tecoGAN
    """)
    #parser.add_argument('--advDiscrThreshold', type=float, default=None, help="""
    #Adverserial training:
    #If the cross entropy loss of the discriminator falls below that threshold, the training for the discriminator is stopped.
    #Set this to zero to disable the check and use a fixed number of iterations, see --advDiscrMaxSteps, instead.
    #""")
    parser_group.add_argument('--advDiscrMaxSteps', type=int, default=1, help="""
    Adverserial training:
    Maximal number of iterations for the discriminator training.
    Set this to -1 to disable the check.
    """)
    parser_group.add_argument('--advDiscrInitialSteps', type=int, default=None, help="""
    Adverserial training:
    Number of iterations for the disciriminator training in the first epoch.
    Used in combination with a pretrained generator to let the discriminator catch up.
    """)
    parser_group.add_argument('--advDiscrWeightClip', type=float, default=0.01, help="""
    For the Wasserstein GAN, this parameter specifies the value of the hyperparameter 'c',
    the range in which the discirminator parameters are clipped.
    """)
    #parser.add_argument('--advGenThreshold', type=float, default=None, help="""
    #Adverserial training:
    #If the cross entropy loss of the generator falls below that threshold, the training for the generator is stopped.
    #Set this to zero to disable the check and use a fixed number of iterations, see --advGenMaxSteps, instead.
    #""")
    parser_group.add_argument('--advGenMaxSteps', type=int, default=1, help="""
    Adverserial training:
    Maximal number of iterations for the generator training.
    Set this to -1 to disable the check.
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
    Strength of the ambient light color in the loss function's shading. Default=0.1
    """)
    parser_group.add_argument('--lossDiffuse', type=float, default=0.1, help="""
    Strength of the diffuse light color in the loss function's shading. Default=1.0
    """)
    parser_group.add_argument('--lossSpecular', type=float, default=0.0, help="""
    Strength of the specular light color in the loss function's shading. Default=0.0
    """)

    parser_group = parser.add_argument_group("Training")
    parser_group.add_argument('--samples', type=int, required=True, help='Number of samples for the train and test dataset')
    parser_group.add_argument('--testFraction', type=float, default=0.2, help='Fraction of test data')
    parser_group.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser_group.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
    parser_group.add_argument('--testNumFullImages', type=int, default=4, help='number of full size images to test for visualization')
    parser_group.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser_group.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser_group.add_argument('--lrGamma', type=float, default=0.5, help='The learning rate decays every lrStep-epochs by this factor')
    parser_group.add_argument('--lrStep', type=int, default=500, help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
    parser_group.add_argument('--weightDecay', type=float, default=0, help="Weight decay (L2 penalty), if supported by the optimizer. Default=0")
    parser_group.add_argument('--optim', type=str, default="Adam", help="""
    Optimizers. Possible values: RMSprop, Rprop, Adam (default).
    """)
    parser_group.add_argument('--noTestImages', action='store_true', help="Don't save full size test images")
    parser_group.add_argument('--cuda', action='store_true', help='use cuda?')
    parser_group.add_argument('--seed', type=int, default=124, help='random seed to use. Default=123')
    parser_group.add_argument('--useCheckpointing', action='store_true', help="""
        Uses checkpointing during training. Checkpointing saves memory, but requires more time.""")

    parser_group = parser.add_argument_group("Output")
    parser_group.add_argument('--logdir', type=str, default='D:/VolumeSuperResolution/logdir_video_3', help='directory for tensorboard logs')
    parser_group.add_argument('--modeldir', type=str, default='D:/VolumeSuperResolution/modeldir_video_3', help='Output directory for the checkpoints')
    parser_group.add_argument('--runName', type=str, help="""
        The name of the run in the output folder.
        If this parameter is omitted, the default name, 'run%05d' with an increasing index is used.""")

    opt = parser.parse_args()
    opt_dict = vars(opt)
    opt_dict['aoInverted'] = False # no ambient occlusion is 0, not 1.
                                  # this flag is passed to the shading as well
    opt_dict['type'] = 'dense2'

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")
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
        nextRunNumber = max(findNextRunNumber(opt.logdir), findNextRunNumber(opt.modeldir)) + 1
        print('Current run: %05d'%nextRunNumber)
        runName = 'run%05d'%nextRunNumber
        logdir = os.path.join(opt.logdir, runName)
        modeldir = os.path.join(opt.modeldir, runName)
        runName = 'run%05d'%nextRunNumber
        os.makedirs(logdir)
        os.makedirs(modeldir)

    #########################
    # DATASETS + CHANNELS
    #########################

    print('===> Loading datasets')

    selectedChannels = opt.inputChannels
    if opt.mode == 'iso':
        selectedChannelsList = (opt.inputChannels or "normal,depth").split(',')
        selectedChannels = [('mask', 0)]
        if "normal" in selectedChannelsList:
            selectedChannels += [('normalX', 1), ('normalY', 2), ('normalZ', 3)]
        if "depth" in selectedChannelsList:
            selectedChannels += [('depth', 4)]
        input_channels = len(selectedChannels) # mask, normal xyz, depth
        output_channels = 6 # mask, normal xyz, depth, ao
    elif opt.mode == 'dvr':
        selectedChannelsList = (opt.inputChannels or "color").split(',')
        selectedChannels = []
        if "color" in selectedChannelsList:
            selectedChannels += [('red', 0), ('green', 1), ('blue', 2), ('alpha', 3)]
        if "normal" in selectedChannelsList:
            selectedChannels += [('normalX', 4), ('normalY', 5), ('normalZ', 6)]
        if "depth" in selectedChannelsList:
            selectedChannels += [('depth', 7)]
        input_channels = len(selectedChannels)
        output_channels = 4 # red green blue alpha
    input_channels_with_previous = input_channels + output_channels * (opt.upscaleFactor ** 2)
    print("selected input channels:", selectedChannels)
    selectedChannels = [t[1] for t in selectedChannels]

    data_augmentation = None
    if opt.mode == 'dvr':
        data_augmentation = dataset.denseDatasetLoaderHDF5_v2.dvr_data_augmentation

    # normalization
    normalizationChannels = [int(c) for c in opt.normalize.split(',')] if opt.normalize else []
    normalizationInfo : dataset.Normalization = \
        dataset.getNormalizationForDataset(opt.targetDataset, 'gt', normalizationChannels)
    normalize = normalizationInfo.getNormalize()
    denormalize = normalizationInfo.getDenormalize()

    train_set = dataset.DenseDatasetFromSamples_v2(
        opt.targetDataset, opt.inputDataset, 
        selectedChannels, output_channels,
        opt.samples, opt.cropSize,
        False, opt.testFraction,
        opt.upscaleFactor,
        40,
        data_augmentation)
    test_set = dataset.DenseDatasetFromSamples_v2(
        opt.targetDataset, opt.inputDataset, 
        selectedChannels, output_channels,
        opt.samples, opt.cropSize,
        True, opt.testFraction,
        opt.upscaleFactor,
        40,
        data_augmentation)
    test_full_set = dataset.DenseDatasetFromSamples_v2(
        opt.targetDataset, opt.inputDataset, 
        selectedChannels, output_channels,
        opt.testNumFullImages, -1,
        False, 0,
        opt.upscaleFactor,
        0,
        data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.numWorkers)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.numWorkers)
    testing_full_data_loader = DataLoader(dataset=test_full_set, batch_size=1, shuffle=False, num_workers=opt.numWorkers)

    #############################
    # MODEL
    #############################

    print('===> Building model')
    ReconOpt = namedtuple("ReconOpt", ["upsample", "reconType", "useBN", "return_residual", "num_layers", "num_channels"])
    reconOpt = ReconOpt(opt.upsample, opt.reconType, opt.useBN, True, opt.numResidualLayers, opt.numFilters)
    model = models.createNetwork(
        opt.model, 
        opt.upscaleFactor,
        input_channels_with_previous, 
        [0,1,2,3,4] if opt.mode=='iso' else [0,1,2,3],
        output_channels,
        reconOpt)
    # apply normalization and color space
    model = models.NormalizedAndColorConvertedNetwork(
        model,
        normalize = normalize, denormalize=denormalize,
        colorSpace = opt.colorSpace if opt.mode == 'dvr' else None)
    model.to(device)
    print('Model:')
    print(model)
    if not no_summary:
        try:
            summary(model, 
                input_size=train_set.get_low_res_shape(input_channels_with_previous), 
                batch_size = 2, device=device.type)
        except RuntimeError as ex:
            print("ERROR in summary writer:", ex)

    # for restoring
    class EnhanceNetWrapper(nn.Module):
        def __init__(self, orig):
            super().__init__();
            self._orig = orig
        def forward(self, input):
            return self._orig(input)

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
        loss_class = losses.LossNetUnshaded
    else:
        loss_class = losses.LossNet
    criterion = loss_class(
        device,
        input_channels,
        output_channels, 
        train_set.get_high_res_shape()[1], #high resolution size
        opt.lossBorderPadding,
        opt.losses,
        opt)
    criterion.to(device)
    if opt.lossesTest is None:
        criterionTest = criterion
    else:
        criterionTest = loss_class(
            device,
            input_channels,
            output_channels, 
            train_set.get_high_res_shape()[1], #high resolution size
            opt.lossBorderPadding,
            opt.losses + "," + opt.lossesTest,
            opt)
        criterionTest.to(device)
    print('Losses:', criterion)
    res = train_set.get_high_res_shape()[1]
    if no_summary:
        criterion.print_summary(
            (output_channels, res, res),
            (output_channels, res, res),
            (input_channels, res, res),
            (output_channels+1, res, res),
            opt.batchSize, device)


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
    if not criterion.has_discriminator:
        adversarial_training = False
        optimizer = createOptimizer(opt.optim, model.parameters(), 
                                    lr=opt.lr, weight_decay=opt.weightDecay)
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lrDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, opt.lrStep, opt.lrGamma)
    else:
        adversarial_training = True
        gen_optimizer = createOptimizer(opt.optim, model.parameters(), 
                                        lr=opt.lr, weight_decay=opt.weightDecay)
        #filter(lambda p: p.requires_grad, criterion.get_discr_parameters())
        discr_optimizer = createOptimizer(
            opt.optim, 
            filter(lambda p: p.requires_grad, criterion.get_discr_parameters()), 
            lr=opt.lr*0.5, weight_decay=opt.weightDecay)
        #gen_scheduler = optim.lr_scheduler.ExponentialLR(gen_optimizer, opt.lrDecay)
        #discr_scheduler = optim.lr_scheduler.ExponentialLR(discr_optimizer, opt.lrDecay)
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, opt.lrStep, opt.lrGamma)
        discr_scheduler = optim.lr_scheduler.StepLR(discr_optimizer, opt.lrStep, opt.lrGamma)

    #############################
    # PRETRAINED
    #############################
    if opt.pretrained is not None:
        checkpoint = torch.load(opt.pretrained)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].state_dict())
        elif 'reconstructionModel' in checkpoint:
            # Import from mainTrainingStepsize or mainTrainingAdaptive
            model._network.load_state_dict(checkpoint['reconstructionModel']._orig.state_dict())
        else:
            raise ValueError("unable to find model in checkpoint")
        #only load the state dict, not the whole model
        #this asserts that the model structure is the same
        print('Using pretrained model for the generator')
    if opt.pretrainedDiscr is not None:
        assert criterion.discriminator is not None
        checkpoint = torch.load(opt.pretrainedDiscr)
        criterion.discriminator.load_state_dict(checkpoint['discriminator'])
        print('Using pretrained model for the discriminator')

    #############################
    # Additional Stuff: Spectral Normalization
    # (placed after pretrained, because models without spectral normalization
    #  can't be imported as models with normalization
    #############################
    if opt.useSN:
        from utils.apply_sn import apply_sn
        apply_sn(model)
        if criterion.discriminator is not None:
            apply_sn(criterion.discriminator)
        print("Spectral Normalization applied")

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
        #model.load_state_dict(checkpoint['state_dict'])
        model = checkpoint['model'] #Restore full model
        if adversarial_training:
            criterion.discriminator.load_state_dict(checkpoint['discriminator'])
            discr_optimizer = checkpoint['discr_optimizer']
            gen_optimizer = checkpoint['gen_optimizer']
            discr_scheduler = checkpoint['discr_scheduler']
            gen_scheduler = checkpoint['gen_scheduler']
        else:
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
        startEpoch = restoreEpoch
        #paths
        print('Current run: %05d'%nextRunNumber)
        runName = 'run%05d'%nextRunNumber
        logdir = os.path.join(opt.logdir, runName)
        modeldir = os.path.join(opt.modeldir, runName)

    optStr = str(opt);
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
        'upscaleFactor', 'cropSize',
        'reconType', 'losses']
    hparams_str = "  \n".join(["%s = %s"%(key, str(opt_dict[key])) for key in important_hparams])
    writer.add_text("hparam", hparams_str)

    #############################
    # MAIN PART
    #############################

    def clampOutput(output):
        if opt.mode == "iso":
            return torch.cat([
                torch.clamp(output[:,0:1,:,:], -1, +1), # mask
                ScreenSpaceShading.normalize(output[:,1:4,:,:], dim=1),
                torch.clamp(output[:,4:5,:,:], 0, +1), # depth
                torch.clamp(output[:,5:6,:,:], 0, +1) # ao
                ], dim=1)
        else: #dvr
            return torch.clamp(output, 0, 1)

    #@profile
    def trainNormal(epoch):
        epoch_loss = 0
        num_minibatch = len(training_data_loader)
        pg = ProgressBar(num_minibatch, 'Training', length=50)
        model.train()
        for iteration, batch in enumerate(training_data_loader, 0):
            pg.print_progress_bar(iteration)
            input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, T, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape
            assert Cout == output_channels
            assert Cin == input_channels, "Cin=%d, input_channels=%d"%(Cin, input_channels)

            optimizer.zero_grad()

            previous_output = None
            loss = 0
            for j in range(T):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, opt.upscaleFactor)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                    previous_warped_loss = target[:,0,:,:,:]
                    previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        opt.upscaleFactor,
                        special_mask = True)
                    previous_warped_loss = previous_warped
                    previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    previous_input = models.VideoTools.warp_upscale(
                        previous_input, 
                        flow[:, j-1, :, :, :], 
                        opt.upscaleFactor,
                        special_mask = True)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscaleFactor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                # run generator
                if opt.useCheckpointing and single_input.requires_grad:
                    prediction, _ = torch.utils.checkpoint.checkpoint(model, single_input)
                else:
                    prediction, _ = model(single_input)
                # evaluate cost
                input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                loss0,_ = criterion(
                    target[:,j,:,:,:], 
                    prediction, 
                    input_high,
                    previous_input,
                    previous_warped_loss,
                    opt.useCheckpointing)
                del _
                loss += loss0
                epoch_loss += loss0.item()
                # save output
                previous_output = clampOutput(prediction)

            loss.backward()
            optimizer.step()
        pg.print_progress_bar(num_minibatch)
        epoch_loss /= num_minibatch * T
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))
        writer.add_scalar('train/total_loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        scheduler.step()
        
    def trainAdv_v2(epoch):
        """
        Second version of adverserial training, 
        for each batch, train both discriminator and generator.
        Not full epoch for each seperately
        """
        print("===> Epoch %d Training"%epoch)

        disc_steps = opt.advDiscrInitialSteps if opt.advDiscrInitialSteps is not None and epoch==1 else opt.advDiscrMaxSteps
        gen_steps = opt.advGenMaxSteps

        num_minibatch = len(training_data_loader)
        model.train()
        criterion.discr_train()

        total_discr_loss = 0
        total_gen_loss = 0
        total_gt_score = 0
        total_pred_score = 0

        pg = ProgressBar(num_minibatch, 'Train', length=50)
        for iteration, batch in enumerate(training_data_loader):
            pg.print_progress_bar(iteration)
            input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, T, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape

            # DISCRIMINATOR
            for _ in range(disc_steps):
                discr_optimizer.zero_grad()
                gen_optimizer.zero_grad()
                loss = 0
                #iterate over all timesteps
                for j in range(T):
                    # prepare input for the generator
                    if j == 0 or opt.disableTemporal:
                        previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, opt.upscaleFactor)
                        # loss takes the ground truth current image as warped previous image,
                        # to not introduce a bias and big loss for the first image
                        previous_warped_loss = target[:,0,:,:,:]
                        previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    else:
                        previous_warped = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                        previous_warped_loss = previous_warped
                        previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                        previous_input = models.VideoTools.warp_upscale(
                            previous_input, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                    previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscaleFactor)
                    single_input = torch.cat((
                            input[:,j,:,:,:],
                            previous_warped_flattened),
                        dim=1)
                    #evaluate generator
                    with torch.no_grad():
                        prediction, _ = model(single_input)
                    #prepare input for the discriminator
                    gt_prev_warped = models.VideoTools.warp_upscale(
                        target[:,j-1,:,:,:], 
                        flow[:, j-1, :, :, :], 
                        opt.upscaleFactor,
                        special_mask = True)
                    #evaluate discriminator
                    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    disc_loss, gt_score, pred_score = criterion.train_discriminator(
                        input_high, 
                        target[:,j,:,:,:], 
                        previous_input, 
                        gt_prev_warped,
                        prediction, 
                        previous_warped_loss,
                        use_checkpoint = False)
                    loss += disc_loss
                    total_gt_score += float(gt_score)
                    total_pred_score += float(pred_score)
                    # save output
                    previous_output = clampOutput(prediction)
                loss.backward()
                discr_optimizer.step()
            total_discr_loss += loss.item()

            # GENERATOR
            for _ in range(disc_steps):
                discr_optimizer.zero_grad()
                gen_optimizer.zero_grad()
                loss = 0
                #iterate over all timesteps
                for j in range(T):
                    # prepare input for the generator
                    if j == 0 or opt.disableTemporal:
                        previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, opt.upscaleFactor)
                        # loss takes the ground truth current image as warped previous image,
                        # to not introduce a bias and big loss for the first image
                        previous_warped_loss = target[:,0,:,:,:]
                        previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    else:
                        previous_warped = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                        previous_warped_loss = previous_warped
                        previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                        previous_input = models.VideoTools.warp_upscale(
                            previous_input, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                    previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscaleFactor)
                    single_input = torch.cat((
                            input[:,j,:,:,:],
                            previous_warped_flattened),
                        dim=1)
                    #evaluate generator
                    if opt.useCheckpointing and single_input.requires_grad:
                        prediction, _ = torch.utils.checkpoint.checkpoint(model, single_input)
                    else:
                        prediction, _ = model(single_input)
                    #evaluate loss
                    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    loss0, map = criterion(
                        target[:,j,:,:,:], 
                        prediction, 
                        input_high,
                        previous_input,
                        previous_warped_loss,
                        opt.useCheckpointing)
                    loss += loss0
                    # save output
                    previous_output = clampOutput(prediction)
                loss.backward()
                gen_optimizer.step()
            total_gen_loss += loss.item()
        pg.print_progress_bar(num_minibatch)

        total_discr_loss /= num_minibatch * T
        total_gen_loss /= num_minibatch * T
        total_gt_score /= num_minibatch * T
        total_pred_score /= num_minibatch * T

        writer.add_scalar('train/discr_loss', total_discr_loss, epoch)
        writer.add_scalar('train/gen_loss', total_gen_loss, epoch)
        writer.add_scalar('train/gt_score', total_gt_score, epoch)
        writer.add_scalar('train/pred_score', total_pred_score, epoch)
        print("===> Epoch {} Complete, gt-score={}, pred-score={}".format(epoch, total_gt_score, total_pred_score))

        discr_scheduler.step()
        writer.add_scalar('train/lr_discr', discr_scheduler.get_lr()[0], epoch)
        gen_scheduler.step()
        writer.add_scalar('train/lr_gen', gen_scheduler.get_lr()[0], epoch)

    #@profile
    def test(epoch):
        avg_psnr = 0
        avg_losses = defaultdict(float)
        with torch.no_grad():
            num_minibatch = len(testing_data_loader)
            pg = ProgressBar(num_minibatch, 'Testing', length=50)
            model.eval()
            if criterion.has_discriminator:
                criterion.discr_eval()
            for iteration, batch in enumerate(testing_data_loader, 0):
                pg.print_progress_bar(iteration)
                input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                B, T, Cout, Hhigh, Whigh = target.shape
                _, _, Cin, H, W = input.shape

                previous_output = None
                for j in range(T):
                    # prepare input
                    if j == 0 or opt.disableTemporal:
                        previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, opt.upscaleFactor)
                        # loss takes the ground truth current image as warped previous image,
                        # to not introduce a bias and big loss for the first image
                        previous_warped_loss = target[:,0,:,:,:]
                        previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    else:
                        previous_warped = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                        previous_warped_loss = previous_warped
                        previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                        previous_input = models.VideoTools.warp_upscale(
                            previous_input, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                    previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscaleFactor)
                    single_input = torch.cat((
                            input[:,j,:,:,:],
                            previous_warped_flattened),
                        dim=1)
                    # run generator
                    prediction, residual = model(single_input)
                    # evaluate cost
                    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    loss0, loss_values = criterionTest(
                        target[:,j,:,:,:], 
                        prediction, 
                        input_high,
                        previous_input,
                        previous_warped_loss)
                    avg_losses['total_loss'] += loss0.item()
                    if opt.mode == "iso":
                        psnr = 10 * log10(1 / max(1e-10, loss_values[('mse','color')]))
                    else:
                        psnr = 10 * log10(1 / max(1e-10, loss_values['mse']))
                    avg_losses['psnr'] += psnr
                    for key, value in loss_values.items():
                        avg_losses[str(key)] += value
                    # save histogram of differences to the input (residual)
                    if iteration==0 and epoch>1:
                        residual = torch.clamp(residual, -0.2, +0.2)
                        residual = residual[:,:,opt.lossBorderPadding:-opt.lossBorderPadding,opt.lossBorderPadding:-opt.lossBorderPadding]
                        if j==0:
                            writer.add_histogram('test/residual-values-t0', residual, epoch, bins='auto')
                        elif j==T//2:
                            writer.add_histogram('test/residual-values-t%d'%j, residual, epoch, bins='auto')
                    ## extra: evaluate discriminator on ground truth data
                    #if criterion.discriminator is not None:
                    #    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    #    if criterion.discriminator_use_previous_image:
                    #        gt_prev_warped = models.VideoTools.warp_upscale(
                    #            target[:,j-1,:,:,:],
                    #            flow[:, j-1, :, :, :], 
                    #            opt.upscaleFactor,
                    #            special_mask = True)
                    #        input_images = torch.cat([input_high, target[:,j,:,:,:], gt_prev_warped], dim=1)
                    #    else:
                    #        input_images = torch.cat([input_high, target[:,j,:,:,:]], dim=1)
                    #    input_images = losses.LossNet.pad(input_images, criterion.padding)
                    #    discr_gt = criterion.adv_loss(criterion.discriminator(input_images))
                    #    avg_losses['discr_gt'] += discr_gt.item()

                    # save output for next frame
                    previous_output = clampOutput(prediction)
            pg.print_progress_bar(num_minibatch)
        for key in avg_losses.keys():
            avg_losses[key] /= num_minibatch * T
        print("===> Avg. PSNR: {:.4f} dB".format(avg_losses['psnr']))
        print("  losses:",avg_losses)
        for key, value in avg_losses.items():
            writer.add_scalar('test/%s'%key, value, epoch)

    def test_images(epoch):
        def write_image(img, filename):
            out_img = img.cpu().detach().numpy()
            out_img *= 255.0
            out_img = out_img.clip(0, 255)
            out_img = np.uint8(out_img)
            writer.add_image(filename, out_img, epoch)

        with torch.no_grad():
            num_minibatch = len(testing_full_data_loader)
            pg = ProgressBar(num_minibatch, 'Test %d Images'%num_minibatch, length=50)
            model.eval()
            if criterion.has_discriminator:
                criterion.discr_eval()
            for i,batch in enumerate(testing_full_data_loader):
                pg.print_progress_bar(i)
                input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                B, T, Cin, H, W = input.shape
                Hhigh = H * opt.upscaleFactor
                Whigh = W * opt.upscaleFactor
                Cout = output_channels

                previous_output = None
                for j in range(T):
                    # prepare input
                    if j == 0 or opt.disableTemporal:
                        previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, opt.upscaleFactor)
                    else:
                        previous_warped = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow[:, j-1, :, :, :], 
                            opt.upscaleFactor,
                            special_mask = True)
                    previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscaleFactor)
                    single_input = torch.cat((
                            input[:,j,:,:,:],
                            previous_warped_flattened),
                        dim=1)
                    # write warped previous frame
                    if opt.mode == 'iso':
                        write_image(previous_warped[0, 1:4], 'image%03d/frame%03d_warped' % (i, j))
                    else:
                        write_image(previous_warped[0, 0:3], 'image%03d/frame%03d_warped' % (i, j))
                    # run generator and cost
                    prediction, residual = model(single_input)
                    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    # save
                    if opt.mode == 'iso':
                        # write prediction
                        prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1)
                        write_image(prediction[0, 1:4], 'image%03d/frame%03d_prediction' % (i, j))
                        if residual is not None:
                            write_image(residual[0, 1:4]+0.5, 'image%03d/frame%03d_residual' % (i, j))
                        # write shaded image
                        shaded_image = shading(prediction)
                        write_image(shaded_image[0], 'image%03d/frame%03d_shaded' % (i, j))
                        # write mask
                        write_image(prediction[0, 0:1, :, :]*0.5+0.5, 'image%03d/frame%03d_mask' % (i, j))
                        # write ambient occlusion
                        write_image(prediction[0, 5:6, :, :], 'image%03d/frame%03d_ao' % (i, j))
                    else: # dvr
                        inputPredGt = torch.cat([
                            input_high[0, 0:3],
                            prediction[0, 0:3],
                            target[0, j, 0:3, :, :]
                            ], dim=2)
                        write_image(inputPredGt, "image%03d/frame%03d_input-pred-gt" % (i,j))
                        if residual is not None:
                            inputPredRes = torch.cat([
                                input_high[0, 0:3],
                                prediction[0, 0:3],
                                residual[0, 0:3] + 0.5
                                ], dim=2)
                            write_image(inputPredRes, "image%03d/frame%03d_input-pred-residual" % (i,j))
                        ##prediction
                        #write_image(prediction[0, 0:3], 'image%03d/frame%03d_prediction' % (i, j))
                        #if residual is not None:
                        #    write_image(residual[0, 0:3]+0.5, 'image%03d/frame%03d_residual' % (i, j))
                        ## write mask
                        #write_image(prediction[0, 3:4, :, :], 'image%03d/frame%03d_mask' % (i, j))
                        ## ground truth
                        #write_image(target[0, j, 0:3, :, :], 'image%03d/frame%03d_gt' % (i, j))
                    # save output for next frame
                    previous_output = clampOutput(prediction)
            pg.print_progress_bar(num_minibatch)

        print("Test images sent to Tensorboard for visualization")

    def checkpoint(epoch):
        model_out_path = os.path.join(modeldir, "model_epoch_{}.pth".format(epoch))
        state = {
            'epoch': epoch + 1, 
            'model': model, 
            'parameters':opt_dict,
            'normalizationInfo':normalizationInfo.getParameters()}
        if not adversarial_training:
            state.update({'optimizer':optimizer, 'scheduler':scheduler})
        else:
            state.update({'discr_optimizer':discr_optimizer, 
                          'gen_optimizer':gen_optimizer,
                          'discr_scheduler':discr_scheduler,
                          'gen_scheduler':gen_scheduler,
                          'criterion': criterion})
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    if not os.path.exists(opt.modeldir):
        os.mkdir(opt.modeldir)
    if not os.path.exists(opt.logdir):
        os.mkdir(opt.logdir)

    print('===> Start Training')
    try:
        if not adversarial_training:
            test(0)
            #test_images(0)
            for epoch in range(startEpoch, opt.nEpochs + 1):
                trainNormal(epoch)
                test(epoch)
                if (epoch < 20 or (epoch%10==0)) and not opt.noTestImages:
                    test_images(epoch)
                if epoch%10==0:
                    checkpoint(epoch)
        else:
            test(0)
            #if not opt.noTestImages:
            #    test_images(0)
            for epoch in range(startEpoch, opt.nEpochs + 1):
                trainAdv_v2(epoch)
                test(epoch)
                if (epoch < 20 or (epoch%10==0)) and not opt.noTestImages:
                    test_images(epoch)
                if epoch%10==0:
                    checkpoint(epoch)

        #writer.export_scalars_to_json(os.path.join(opt.logdir, "all_scalars.json"))
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