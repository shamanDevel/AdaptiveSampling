from __future__ import print_function
import argparse
import math
from math import log10
import os
import os.path
from collections import defaultdict
import itertools

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
from utils import ScreenSpaceShading, initialImage

# Training settings
parser = argparse.ArgumentParser(description='Superresolution for Direct Volume Rendering')

parser.add_argument('--dataset', type=str,
                    help="Path to the HDF5 file with the dataset", required=True)
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--numberOfImages', type=int, default=-1, help="Number of images taken from the inpt dataset. Default: -1 = unlimited")

parser.add_argument('--restore', type=int, default=-1, help="Restore training from the specified run index")
parser.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
parser.add_argument('--pretrained', type=str, default=None, help="Path to a pretrained generator")
parser.add_argument('--pretrainedDiscr', type=str, default=None, help="Path to a pretrained discriminator")

#Model parameters
parser.add_argument('--model', type=str, required=True, help="""
The superresolution model.
Supported nets: 'SubpixelNet', 'EnhanceNet', 'TecoGAN', 'RCAN'
""")
parser.add_argument('--upsample', type=str, default='bilinear', help='Upsampling for EnhanceNet: nearest, bilinear, bicubic, or pixelShuffle')
parser.add_argument('--reconType', type=str, default='residual', help='Block type for EnhanceNet: residual or direct')
parser.add_argument('--useBN', action='store_true', help='Enable batch normalization in the generator and discriminator')
parser.add_argument('--useSN', action='store_true', help='Enable spectral normalization in the generator and discriminator')
parser.add_argument('--numResidualLayers', type=int, default=10, help='Number of residual layers in the generator')
parser.add_argument('--disableTemporal', action='store_true', help='Disables temporal consistency')
parser.add_argument('--initialImage', type=str, default='zero', help="""
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
parser.add_argument('--losses', type=str, required=True, help="""
Comma-separated list of loss functions: mse,perceptual,texture,adv. 
Optinally, the weighting factor can be specified with a colon.
Example: "--losses perceptual:0.1,texture:1e2,adv:10"
""")
parser.add_argument('--perceptualLossLayers', 
                    type=str, 
                     # defaults found with VGGAnalysis.py
                    default='conv_1:0.026423,conv_2:0.009285,conv_3:0.006710,conv_4:0.004898,conv_5:0.003910,conv_6:0.003956,conv_7:0.003813,conv_8:0.002968,conv_9:0.002997,conv_10:0.003631,conv_11:0.004147,conv_12:0.005765,conv_13:0.007442,conv_14:0.009666,conv_15:0.012586,conv_16:0.013377', 
                    help="""
Comma-separated list of layer names for the perceptual loss. 
Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
Optionally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
""")
parser.add_argument('--textureLossLayers', type=str, default='conv_1,conv_3,conv_5', help="""
Comma-separated list of layer names for the perceptual loss. 
Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
Optinally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
""")
parser.add_argument('--discriminator', type=str, default='enhanceNetLarge', help="""
Network architecture for the discriminator.
Possible values: enhanceNetSmall, enhanceNetLarge, tecoGAN
""")
#parser.add_argument('--advDiscrThreshold', type=float, default=None, help="""
#Adverserial training:
#If the cross entropy loss of the discriminator falls below that threshold, the training for the discriminator is stopped.
#Set this to zero to disable the check and use a fixed number of iterations, see --advDiscrMaxSteps, instead.
#""")
parser.add_argument('--advDiscrMaxSteps', type=int, default=2, help="""
Adverserial training:
Maximal number of iterations for the discriminator training.
Set this to -1 to disable the check.
""")
parser.add_argument('--advDiscrInitialSteps', type=int, default=None, help="""
Adverserial training:
Number of iterations for the disciriminator training in the first epoch.
Used in combination with a pretrained generator to let the discriminator catch up.
""")
parser.add_argument('--advDiscrWeightClip', type=float, default=0.01, help="""
For the Wasserstein GAN, this parameter specifies the value of the hyperparameter 'c',
the range in which the discirminator parameters are clipped.
""")
#parser.add_argument('--advGenThreshold', type=float, default=None, help="""
#Adverserial training:
#If the cross entropy loss of the generator falls below that threshold, the training for the generator is stopped.
#Set this to zero to disable the check and use a fixed number of iterations, see --advGenMaxSteps, instead.
#""")
parser.add_argument('--advGenMaxSteps', type=int, default=2, help="""
Adverserial training:
Maximal number of iterations for the generator training.
Set this to -1 to disable the check.
""")
parser.add_argument('--lossBorderPadding', type=int, default=16, help="""
Because flow + warping can't be accurately estimated at the borders of the image,
the border of the input images to the loss (ground truth, low res input, prediction)
are overwritten with zeros. The size of the border is specified by this parameter.
Pass zero to disable this padding. Default=16 as in the TecoGAN paper.
""")

parser.add_argument('--samples', type=int, required=True, help='Number of samples for the train and test dataset')
parser.add_argument('--testFraction', type=float, default=0.2, help='Fraction of test data')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--testNumFullImages', type=int, default=4, help='number of full size images to test for visualization')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--lrGamma', type=float, default=0.5, help='The learning rate decays every lrStep-epochs by this factor')
parser.add_argument('--lrStep', type=int, default=500, help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
parser.add_argument('--weightDecay', type=float, default=0, help="Weight decay (L2 penalty), if supported by the optimizer. Default=0")
parser.add_argument('--optim', type=str, default="Adam", help="""
Optimizers. Possible values: RMSprop, Rprop, Adam (default).
""")
parser.add_argument('--noTestImages', action='store_true', help="Don't save full size test images")

parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')
parser.add_argument('--logdir', type=str, default='C:/Users/Mustafa/Desktop/dvrRendering/log', help='directory for tensorboard logs')
parser.add_argument('--modeldir', type=str, default='C:/Users/Mustafa/Desktop/dvrRendering/model', help='Output directory for the checkpoints')

opt = parser.parse_args()
opt_dict = vars(opt)

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
nextRunNumber = max(findNextRunNumber(opt.logdir), findNextRunNumber(opt.modeldir)) + 1
if opt.restore == -1:
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
dataset_data = dataset.dvr_dense_load_samples_hdf5(opt_dict['upscale_factor'], opt_dict, opt_dict['dataset'])
print('Dataset input images have %d channels'%dataset_data.input_channels)

input_channels = dataset_data.input_channels
assert input_channels == 4 # RGB, MASK
output_channels = dataset_data.output_channels
assert output_channels == 4 # RGB, MASK
input_channels_with_previous = input_channels + output_channels * (opt.upscale_factor ** 2)

train_set = dataset.DvrDenseDatasetFromSamples(dataset_data, False, opt.testFraction, device)
test_set = dataset.DvrDenseDatasetFromSamples(dataset_data, True, opt.testFraction, device)
test_full_set = dataset.DvrDenseDatasetFromFullImages(dataset_data, min(opt.testNumFullImages, len(dataset_data.images_low)))
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)
testing_full_data_loader = DataLoader(dataset=test_full_set, batch_size=1, shuffle=False)

#############################
# MODEL
#############################
print('===> Building model')
model = models.createNetwork(
    opt.model, 
    opt.upscale_factor,
    input_channels_with_previous, 
    [0,1,2,3],
    output_channels,
    opt)
model.to(device)
print('Model:')
print(model)
if not no_summary:
    summary(model, 
            input_size=train_set.get_low_res_shape(input_channels_with_previous), 
            device=device.type)

#############################
# LOSSES
#############################

print('===> Building losses')
criterion = losses.LossNet(
    device,
    input_channels,
    output_channels, 
    train_set.get_high_res_shape()[1], #high resolution size
    opt.lossBorderPadding,
    opt)
criterion.to(device)
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
    model.load_state_dict(checkpoint['model'].state_dict())
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
    text_file.write(optStr)

#tensorboard logger
writer = SummaryWriter(logdir)
writer.add_text('info', optStr, 0)

#############################
# MAIN PART
#############################

#@profile
def trainNormal(epoch):
    epoch_loss = 0
    num_minibatch = len(training_data_loader)
    pg = ProgressBar(num_minibatch, 'Training', length=50)
    model.train()
    for iteration, batch in enumerate(training_data_loader, 0):
        pg.print_progress_bar(iteration)
        input, target = batch[0].to(device), batch[1].to(device)
        B, _, Cout, Hhigh, Whigh = target.shape
        _, _, Cin, H, W = input.shape
        assert(Cout == output_channels)
        assert(Cin == input_channels)
        assert(H == dataset_data.crop_size)
        assert(W == dataset_data.crop_size)
        assert(Hhigh == dataset_data.crop_size * opt.upscale_factor)
        assert(Whigh == dataset_data.crop_size * opt.upscale_factor)

        optimizer.zero_grad()

        previous_output = None
        loss = 0
        for j in range(dataset_data.num_frames):
            # prepare input
            if j == 0 or opt.disableTemporal:
                previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                               opt.initialImage, False, opt.upscale_factor)
                # loss takes the ground truth current image as warped previous image,
                # to not introduce a bias and big loss for the first image
                previous_warped_loss = target[:,0,:,:,:]
                previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
            else:
                previous_warped = models.VideoTools.warp_upscale(
                    previous_output, 
                    flow[:, j-1, :, :, :], 
                    opt.upscale_factor,
                    special_mask = True)
                previous_warped_loss = previous_warped
                previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                previous_input = models.VideoTools.warp_upscale(
                    previous_input, 
                    flow[:, j-1, :, :, :], 
                    opt.upscale_factor,
                    special_mask = True)
            previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
            single_input = torch.cat((
                    input[:,j,:,:,:],
                    previous_warped_flattened),
                dim=1)
            # run generator
            prediction, _ = model(single_input)
            # evaluate cost
            input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
            loss0,_ = criterion(
                target[:,j,:,:,:], 
                prediction, 
                input_high,
                previous_warped_loss)
            del _
            loss += loss0
            epoch_loss += loss0.item()
            # save output
            previous_output = prediction

        loss.backward()
        optimizer.step()
    pg.print_progress_bar(num_minibatch)
    epoch_loss /= num_minibatch * dataset_data.num_frames
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
    discr_scheduler.step()
    writer.add_scalar('train/lr_discr', discr_scheduler.get_lr()[0], epoch)
    gen_scheduler.step()
    writer.add_scalar('train/lr_gen', gen_scheduler.get_lr()[0], epoch)

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
        B, _, Cout, Hhigh, Whigh = target.shape
        _, _, Cin, H, W = input.shape

        # DISCRIMINATOR
        for _ in range(disc_steps):
            discr_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            loss = 0
            #iterate over all timesteps
            for j in range(dataset_data.num_frames):
                # prepare input for the generator
                if j == 0 or opt.disableTemporal:
                    previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                               opt.initialImage, False, opt.upscale_factor)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                    previous_warped_loss = target[:,0,:,:,:]
                    previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                    previous_warped_loss = previous_warped
                    previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    previous_input = models.VideoTools.warp_upscale(
                        previous_input, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
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
                    opt.upscale_factor,
                    special_mask = True)
                #evaluate discriminator
                input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                disc_loss, gt_score, pred_score = criterion.train_discriminator(
                    input_high, 
                    target[:,j,:,:,:], 
                    previous_input, 
                    gt_prev_warped,
                    prediction, 
                    previous_warped_loss)
                loss += disc_loss
                total_gt_score += float(gt_score)
                total_pred_score += float(pred_score)
                # save output
                previous_output = torch.cat([
                    torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                    ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                    torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                    torch.clamp(prediction[:,5:6,:,:], 0, +1) # ao
                    ], dim=1)
            loss.backward()
            discr_optimizer.step()
        total_discr_loss += loss.item()

        # GENERATOR
        for _ in range(disc_steps):
            discr_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            loss = 0
            #iterate over all timesteps
            for j in range(dataset_data.num_frames):
                # prepare input for the generator
                if j == 0 or opt.disableTemporal:
                    previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                               opt.initialImage, False, opt.upscale_factor)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                    previous_warped_loss = target[:,0,:,:,:]
                    previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                    previous_warped_loss = previous_warped
                    previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    previous_input = models.VideoTools.warp_upscale(
                        previous_input, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                #evaluate generator
                prediction, _ = model(single_input)
                #evaluate loss
                input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                loss0, map = criterion(
                    target[:,j,:,:,:], 
                    prediction, 
                    input_high,
                    previous_input,
                    previous_warped_loss)
                loss += loss0
                # save output
                previous_output = torch.cat([
                    torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                    ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                    torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                    torch.clamp(prediction[:,5:6,:,:], 0, +1) # ao
                    ], dim=1)
            loss.backward()
            gen_optimizer.step()
        total_gen_loss += loss.item()
    pg.print_progress_bar(num_minibatch)

    total_discr_loss /= num_minibatch * dataset_data.num_frames
    total_gen_loss /= num_minibatch * dataset_data.num_frames
    total_gt_score /= num_minibatch * dataset_data.num_frames
    total_pred_score /= num_minibatch * dataset_data.num_frames

    writer.add_scalar('train/discr_loss', total_discr_loss, epoch)
    writer.add_scalar('train/gen_loss', total_gen_loss, epoch)
    writer.add_scalar('train/gt_score', total_gt_score, epoch)
    writer.add_scalar('train/pred_score', total_pred_score, epoch)
    print("===> Epoch {} Complete".format(epoch))

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
            input, target = batch[0].to(device), batch[1].to(device)
            B, _, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape

            previous_output = None
            for j in range(dataset_data.num_frames):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                               opt.initialImage, False, opt.upscale_factor)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                    previous_warped_loss = target[:,0,:,:,:]
                    previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                    previous_warped_loss = previous_warped
                    previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    previous_input = models.VideoTools.warp_upscale(
                        previous_input, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                # run generator
                prediction, _ = model(single_input)
                # evaluate cost
                input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                loss0, loss_values = criterion(
                    target[:,j,:,:,:], 
                    prediction, 
                    input_high,
                    previous_warped_loss)
                avg_losses['total_loss'] += loss0.item()
                psnr = 10 * log10(1 / max(1e-10, loss_values['mse']))
                avg_losses['psnr'] += psnr
                for key, value in loss_values.items():
                    avg_losses[str(key)] += value

                # save output for next frame
                previous_output = prediction
        pg.print_progress_bar(num_minibatch)
    for key in avg_losses.keys():
        avg_losses[key] /= num_minibatch * dataset_data.num_frames
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
            input = batch[0].to(device)
            B, _, Cin, H, W = input.shape
            Hhigh = H * opt.upscale_factor
            Whigh = W * opt.upscale_factor
            Cout = output_channels

            channel_mask = [0, 1, 2] #RGB

            previous_output = None
            for j in range(dataset_data.num_frames):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                               opt.initialImage, False, opt.upscale_factor)
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        opt.upscale_factor,
                        special_mask = True)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                # run generator and cost
                prediction, residual = model(single_input)
                # write prediction image
                write_image(prediction[0, channel_mask], 'image%03d/frame%03d_prediction' % (i, j))
                # write residual image
                if residual is not None:
                    write_image(residual[0, channel_mask], 'image%03d/frame%03d_residual' % (i, j))
                # save output for next frame
                previous_output = prediction
        pg.print_progress_bar(num_minibatch)

    print("Test images sent to Tensorboard for visualization")

def checkpoint(epoch):
    model_out_path = os.path.join(modeldir, "model_epoch_{}.pth".format(epoch))
    state = {'epoch': epoch + 1, 'model': model, 'parameters':opt_dict}
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
if not adversarial_training:
    #test_images(0)
    for epoch in range(startEpoch, opt.nEpochs + 1):
        trainNormal(epoch)
        test(epoch)
        if (epoch < 20 or (epoch%10==0)) and not opt.noTestImages:
            test_images(epoch)
        checkpoint(epoch)
else:
    test(0)
    if not opt.noTestImages:
        test_images(0)
    for epoch in range(startEpoch, opt.nEpochs + 1):
        trainAdv_v2(epoch)
        test(epoch)
        if (epoch < 20 or (epoch%10==0)) and not opt.noTestImages:
            test_images(epoch)
        if epoch%10==0:
            checkpoint(epoch)

#writer.export_scalars_to_json(os.path.join(opt.logdir, "all_scalars.json"))
writer.close()