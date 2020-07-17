"""
This trains importance.NetworkImportanceMap, a network that generates
from a low-resolution input image a heat map that is then used to generate
samples for a high-resolution rendering
"""

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

import dataset
import models
import losses
import importance
from utils import ScreenSpaceShading, initialImage

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(
        description='Importance Map generation for adaptive sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_group = parser.add_argument_group("Dataset")
    parser_group.add_argument('--dataset', type=str, required=True,
                        help="Path the the HDF5 file with the dataset")
    parser_group.add_argument('--numberOfImages', type=int, default=-1, help="Number of images taken from the inpt dataset. Default: -1 = unlimited")
    parser_group.add_argument('--samples', type=int, required=True, help='Number of samples for the train and test dataset')
    parser_group.add_argument('--testFraction', type=float, default=0.2, help='Fraction of test data')

    parser_group = parser.add_argument_group("Restore")
    parser_group.add_argument('--restore', type=int, default=-1, help="Restore training from the specified run index")
    parser_group.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
    parser_group.add_argument('--pretrained', type=str, default=None, help="Path to a pretrained generator")
    parser_group.add_argument('--pretrainedDiscr', type=str, default=None, help="Path to a pretrained discriminator")

    #Model parameters
    parser_group = parser.add_argument_group("Model")
    parser_group.add_argument('--networkUpscale', type=int, default=4, help="upscale factor of the importance map network")
    parser_group.add_argument('--model', type=str, default="EnhanceNet", help="""
    The superresolution model, support is only 'EnhanceNet' so far.
    """)
    parser_group.add_argument('--useBN', action='store_true', help='Enable batch normalization in the generator and discriminator')
    parser_group.add_argument('--border', type=int, default=8, help='Zero border around the network')
    parser_group.add_argument('--outputLayer', type=str, default='softplus',
                        help="Network output layer, either 'none', 'softplus' or 'sigmoid'")
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
    parser_group.add_argument('--minImportance', type=float, default=0.01,
                              help="the minimal importance value, i.e. the maximal spacing of samples")
    parser_group.add_argument('--meanImportance', type=float, default=0.2,
                              help="the mean importance value, i.e. the average number of samples")
    parser_group.add_argument('--postUpscale', type=int, default=1, 
                        help="Upscaling factor applied after the network for a smoother map")
    parser_group.add_argument('--distanceToStandardDeviation', type=float, default=0.5)

    #Loss parameters
    parser_group = parser.add_argument_group("Loss")
    parser_group.add_argument('--losses', type=str, required=True, help="""
    Comma-separated list of loss functions: mse,perceptual,texture,adv. 
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
    Strength of the ambient light color in the loss function's shading. Default=0.1
    """)
    parser_group.add_argument('--lossDiffuse', type=float, default=0.1, help="""
    Strength of the diffuse light color in the loss function's shading. Default=1.0
    """)
    parser_group.add_argument('--lossSpecular', type=float, default=0.0, help="""
    Strength of the specular light color in the loss function's shading. Default=0.0
    """)
    parser_group.add_argument('--lossHeatmapMean', type=float, default=0.1, 
                              help="Loss weight of the additional loss term that forces the heatmap to have mean 0.5")

    parser_group = parser.add_argument_group("Training")
    parser_group.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser_group.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
    parser_group.add_argument('--testNumFullImages', type=int, default=8, help='number of full size images to test for visualization')
    parser_group.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser_group.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser_group.add_argument('--lrGamma', type=float, default=0.5, help='The learning rate decays every lrStep-epochs by this factor')
    parser_group.add_argument('--lrStep', type=int, default=500, help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
    parser_group.add_argument('--weightDecay', type=float, default=0, help="Weight decay (L2 penalty), if supported by the optimizer. Default=0")
    parser_group.add_argument('--optim', type=str, default="Adam", help="""
    Optimizers. Possible values: RMSprop, Rprop, Adam (default).
    """)
    parser_group.add_argument('--noTestImages', action='store_true', help="Don't save full size test images")
    parser_group.add_argument('--noCuda', action='store_true', help='Disable cuda')
    parser_group.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')

    parser_group = parser.add_argument_group("Output")
    parser_group.add_argument('--logdir', type=str, default='D:/VolumeSuperResolution/importance_logdir', help='directory for tensorboard logs')
    parser_group.add_argument('--modeldir', type=str, default='D:/VolumeSuperResolution/importance_modeldir', help='Output directory for the checkpoints')

    opt = parser.parse_args()
    opt_dict = vars(opt)
    opt_dict['type'] = 'importance1'
    opt_dict['aoInverted'] = False # no ambient occlusion is 0, not 1.
                                  # this flag is passed to the shading as well

    if not opt.noCuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --noCuda")
    device = torch.device("cpu" if opt.noCuda else "cuda")

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.set_num_threads(4)

    rendererPath = "./Renderer.dll"
    if not os.path.isfile(rendererPath):
        rendererPath = '../bin/Renderer.dll'
    if not os.path.isfile(rendererPath):
        raise ValueError("Unable to locate Renderer.dll")
    rendererPath = os.path.abspath(rendererPath)
    print("Load renderer library:", rendererPath)
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(rendererPath))
    torch.ops.load_library('./Renderer.dll')
    os.chdir(old_cwd)

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
    upscale_factor = opt_dict['networkUpscale']*opt_dict['postUpscale']
    dataset_data = dataset.dense_load_samples_hdf5(
        upscale_factor, 
        opt_dict, 
        opt_dict['dataset'])
    print('Dataset input images have %d channels'%dataset_data.input_channels)

    input_channels = dataset_data.input_channels
    assert input_channels == 5 # mask, normalX, normalY, normalZ, depth
    output_channels = dataset_data.output_channels
    assert output_channels == 1 # importance
    input_channels_with_previous = input_channels + output_channels * (upscale_factor ** 2)

    train_set = dataset.DenseDatasetFromSamples(dataset_data, False, opt.testFraction, device)
    test_set = dataset.DenseDatasetFromSamples(dataset_data, True, opt.testFraction, device)
    test_full_set = dataset.DenseDatasetFromFullImages(dataset_data, min(opt.testNumFullImages, len(dataset_data.images_low)))
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)
    testing_full_data_loader = DataLoader(dataset=test_full_set, batch_size=1, shuffle=False)

    #############################
    # MODEL
    #############################

    #TODO: temporal component

    print('===> Building model')
    model = importance.NetworkImportanceMap(
        opt.networkUpscale, 
        input_channels,
        model = opt.model,
        use_bn = opt.useBN,
        border_padding = opt.border,
        output_layer = opt.outputLayer)
    postprocess = importance.PostProcess(
            opt.minImportance, opt.meanImportance, 
            opt.postUpscale,
            opt.lossBorderPadding // opt.postUpscale)
    model.to(device)
    print('Model:')
    print(model)
    if not no_summary:
        summary(model, 
                input_size=train_set.get_low_res_shape(input_channels), 
                device=device.type)

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
    criterion = losses.LossNetUnshaded(
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
        raise ValueError("Adverserial training not longer supported")

    #############################
    # PRETRAINED
    #############################
    if opt.pretrained is not None:
        checkpoint = torch.load(opt.pretrained)
        model.load_state_dict(checkpoint['model'].state_dict())
        #only load the state dict, not the whole model
        #this asserts that the model structure is the same
        print('Using pretrained model for the generator')

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
            input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, _, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape
            assert(Cout == output_channels)
            assert(Cin == input_channels)
            assert(H == dataset_data.crop_size)
            assert(W == dataset_data.crop_size)
            assert(Hhigh == dataset_data.crop_size * upscale_factor)
            assert(Whigh == dataset_data.crop_size * upscale_factor)

            optimizer.zero_grad()

            previous_output = None
            loss = 0
            for j in range(1):#range(dataset_data.num_frames):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, upscale_factor)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                    previous_warped_loss = target[:,0,:,:,:]
                    previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode='bilinear')
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        upscale_factor,
                        special_mask = True)
                    previous_warped_loss = previous_warped
                    previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode='bilinear')
                    previous_input = models.VideoTools.warp_upscale(
                        previous_input, 
                        flow[:, j-1, :, :, :], 
                        upscale_factor,
                        special_mask = True)
                # TODO: enable temporal component again
                #previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                #single_input = torch.cat((
                #        input[:,j,:,:,:],
                #        previous_warped_flattened),
                #    dim=1)
                single_input = input[:,j,:,:,:]
                # run generator
                heatMap = model(single_input)
                heatMapCrop = heatMap[:,opt.lossBorderPadding:-opt.lossBorderPadding,opt.lossBorderPadding:-opt.lossBorderPadding]
                heatMap = postprocess(heatMap)
                prediction = importance.adaptiveSmoothing(
                    target[:,j,:,:,:].contiguous(), 1/heatMap.unsqueeze(1),
                    opt.distanceToStandardDeviation)
                # evaluate cost
                input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode='bilinear')
                loss0,_ = criterion(
                    target[:,j,:,:,:], 
                    prediction, 
                    input_high,
                    previous_input,
                    previous_warped_loss)
                del _
                loss0 += opt.lossHeatmapMean * ((0.5-torch.mean(heatMapCrop))**2)
                #print("Mean:",torch.mean(heatMapCrop).item())
                loss += loss0
                epoch_loss += loss0.item()
                # save output
                previous_output = prediction

            #loss.retain_grad()
            loss.backward()
            optimizer.step()
        pg.print_progress_bar(num_minibatch)
        epoch_loss /= num_minibatch * dataset_data.num_frames
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))
        writer.add_scalar('train/total_loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        scheduler.step()
   
    #@profile
    def test(epoch):
        avg_psnr = 0
        avg_losses = defaultdict(float)
        heatmap_min = 1e10
        heatmap_max = -1e10
        heatmap_avg = heatmap_count = 0
        with torch.no_grad():
            num_minibatch = len(testing_data_loader)
            pg = ProgressBar(num_minibatch, 'Testing', length=50)
            model.eval()
            if criterion.has_discriminator:
                criterion.discr_eval()
            for iteration, batch in enumerate(testing_data_loader, 0):
                pg.print_progress_bar(iteration)
                input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                B, _, Cout, Hhigh, Whigh = target.shape
                _, _, Cin, H, W = input.shape

                previous_output = None
                for j in range(dataset_data.num_frames):
                    # prepare input
                    if j == 0 or opt.disableTemporal:
                        previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, upscale_factor)
                        # loss takes the ground truth current image as warped previous image,
                        # to not introduce a bias and big loss for the first image
                        previous_warped_loss = target[:,0,:,:,:]
                        previous_input = F.interpolate(input[:,0,:,:,:], size=(Hhigh, Whigh), mode='bilinear')
                    else:
                        previous_warped = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow[:, j-1, :, :, :], 
                            upscale_factor,
                            special_mask = True)
                        previous_warped_loss = previous_warped
                        previous_input = F.interpolate(input[:,j-1,:,:,:], size=(Hhigh, Whigh), mode='bilinear')
                        previous_input = models.VideoTools.warp_upscale(
                            previous_input, 
                            flow[:, j-1, :, :, :], 
                            upscale_factor,
                            special_mask = True)
                    # TODO: enable temporal component again
                    #previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                    #single_input = torch.cat((
                    #        input[:,j,:,:,:],
                    #        previous_warped_flattened),
                    #    dim=1)
                    single_input = input[:,j,:,:,:]
                    # run generator
                    heatMap = model(single_input)
                    heatMapCrop = heatMap[:,opt.lossBorderPadding:-opt.lossBorderPadding,opt.lossBorderPadding:-opt.lossBorderPadding]
                    heatmap_min = min(heatmap_min, torch.min(heatMapCrop).item())
                    heatmap_max = max(heatmap_max, torch.max(heatMapCrop).item())
                    heatmap_avg += torch.mean(heatMapCrop).item()
                    heatmap_count += 1
                    heatMap = postprocess(heatMap)
                    prediction = importance.adaptiveSmoothing(
                        target[:,j,:,:,:].contiguous(), 1/heatMap.unsqueeze(1),
                        opt.distanceToStandardDeviation)
                    # evaluate cost
                    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode='bilinear')
                    loss0, loss_values = criterion(
                        target[:,j,:,:,:], 
                        prediction, 
                        input_high,
                        previous_input,
                        previous_warped_loss)
                    avg_losses['total_loss'] += loss0.item()
                    psnr = 10 * log10(1 / max(1e-10, loss_values[('mse','color')]))
                    avg_losses['psnr'] += psnr
                    for key, value in loss_values.items():
                        avg_losses[str(key)] += value

                    # save output for next frame
                    previous_output = torch.cat([
                        torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                        ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                        torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                        torch.clamp(prediction[:,5:6,:,:], 0, +1) # ao
                        ], dim=1)
            pg.print_progress_bar(num_minibatch)
        for key in avg_losses.keys():
            avg_losses[key] /= num_minibatch * dataset_data.num_frames
        print("===> Avg. PSNR: {:.4f} dB".format(avg_losses['psnr']))
        print("  losses:",avg_losses)
        for key, value in avg_losses.items():
            writer.add_scalar('test/%s'%key, value, epoch)
        print("  heatmap: min=%f, max=%f, avg=%f"%(heatmap_min, heatmap_max, heatmap_avg/heatmap_count))
        writer.add_scalar('test/heatmap_min', heatmap_min, epoch)
        writer.add_scalar('test/heatmap_max', heatmap_max, epoch)
        writer.add_scalar('test/heatmap_avg', heatmap_avg/heatmap_count, epoch)

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
                B, _, Cin, H, W = input.shape
                Hhigh = H * upscale_factor
                Whigh = W * upscale_factor
                Cout = output_channels

                channel_mask = [1, 2, 3] #normal

                previous_output = None
                for j in range(dataset_data.num_frames):
                    # prepare input
                    if j == 0 or opt.disableTemporal:
                        previous_warped = initialImage(input[:,0,:,:,:], Cout, 
                                                   opt.initialImage, False, upscale_factor)
                    else:
                        previous_warped = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow[:, j-1, :, :, :], 
                            upscale_factor,
                            special_mask = True)
                    # TODO: enable temporal component again
                    #previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                    #single_input = torch.cat((
                    #        input[:,j,:,:,:],
                    #        previous_warped_flattened),
                    #    dim=1)
                    single_input = input[:,j,:,:,:]
                    # run generator
                    heatMap = model(single_input)
                    heatMap = postprocess(heatMap)
                    prediction = importance.adaptiveSmoothing(
                        target[:,j,:,:,:].contiguous(), 1/heatMap.unsqueeze(1),
                        opt.distanceToStandardDeviation)
                    # write heatmap
                    write_image(heatMap[0].unsqueeze(0), 'image%03d/frame%03d_heatmap' % (i, j))
                    ## write warped previous frame
                    #write_image(previous_warped[0, channel_mask], 'image%03d/frame%03d_warped' % (i, j))
                    # write predicted normals
                    prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1)
                    write_image(prediction[0, channel_mask], 'image%03d/frame%03d_prediction' % (i, j))
                    # write shaded image if network runs in deferredShading mode
                    shaded_image = shading(prediction)
                    write_image(shaded_image[0], 'image%03d/frame%03d_shaded' % (i, j))
                    # write mask
                    write_image(prediction[0, 0:1, :, :]*0.5+0.5, 'image%03d/frame%03d_mask' % (i, j))
                    # write ambient occlusion
                    write_image(prediction[0, 5:6, :, :], 'image%03d/frame%03d_ao' % (i, j))
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
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #    trainNormal(epoch)
            #print("Export profiler results")
            #prof.export_chrome_trace(os.path.join(modeldir, 'profile_%d.json'%epoch))
            #print(prof.key_averages().table(sort_by="self_cpu_time_total"))

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