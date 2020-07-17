import warnings
warnings.filterwarnings('ignore')

import argparse
import math
from math import log10
import os
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np
from console_progressbar import ProgressBar

import dataset
import models
import losses
import utils

if __name__ == "__main__":

    no_summary = False
    try:
        from torchsummary import summary
    except ModuleNotFoundError:
        no_summary = True
        print("No summary writer found")

    #########################################################
    # ARGUMENT PARSING
    #########################################################
    parser = argparse.ArgumentParser(
        description='Superresolution for Isosurface Raytracing - Sparse',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser_group = parser.add_argument_group("Dataset")
    parser_group.add_argument('--dataset', type=str, default=None, help="""
        Semicolon-separated list of directories with the dataset of numpy images.\n
        Specify either '--dataset' or '--hdf5'""")
    parser_group.add_argument('--hdf5', type=str, default=None, help="""
        Semicolon-separated list of HDF5 files with the dataset.
        The HDF5 files are created with a specific crop size, so all crop settings are ignored.
        Specify either '--dataset' or '--hdf5'""")
    parser_group.add_argument('--numberOfImages', type=int, default=-1, 
                              help="Number of images taken from the inpt dataset. Default: -1 = unlimited. Ignored for HDF5-dataset")
    parser_group.add_argument('--testFraction', type=float, default=0.2, 
                              help="The fraction of samples used for testing")
    parser_group.add_argument('--trainCropsPerImage', type=int, default=51, 
                              help="The number of crops per image. Ignored for HDF5-dataset")
    parser_group.add_argument('--testCropsPerImage', type=int, default=23, 
                              help="The number of crops per image. Ignored for HDF5-dataset")
    parser_group.add_argument('--trainCropSize', type=int, default=64, 
                              help="The size of the crops used for training. Ignored for HDF5-dataset")
    parser_group.add_argument('--testCropSize', type=int, default=256, 
                              help="The size of the crops used for testing. Ignored for HDF5-dataset")
    parser_group.add_argument('--interpolateInput', action='store_true', help=
        """Use interpolated input from the sparse samples as input instead of only the samples.
        This will append an extra channel to the network input with the plain sample mask.""")
    parser_group.add_argument('--bufferSize', type=int, default=5, 
                              help="The number of images that are loaded and buffered asynchronously. Ignored for HDF5-dataset")
    parser_group.add_argument('--cropFillRate', type=float, default=0.4,
                              help="crops are only taken if at least cropFillRate*100%% pixels in the dense image are filled. Ignored for HDF5-dataset")
    parser_group.add_argument('--smoothDense', action='store_true',
                              help="Set to true to use smoothed target images")
    parser_group.add_argument('--externalFlow', action='store_true', help="""
        False (old, default): the network gets the flow also as input an has to estimate it. The warping is done with the previous flow.
        True (new): the optical flow for the warping is taken from the interpolated sparse samples and not passed to the network.
    """)

    # Model
    parser_group = parser.add_argument_group("Model")
    parser_group.add_argument('--architecture', type=str, default='UNet', choices=['UNet', 'DeepFovea'],
                              help="The network architecture, supported are 'UNet' and 'DeepFovea'")
    parser_group.add_argument('--depth', type=int, default=6, help="The depth of the network")
    #parser_group.add_argument('--filters', type=int, default=6, nargs='+', help=""" 
    parser_group.add_argument('--filters', type=int, default=6, help=""" 
            UNet: an int, the number of filters in the first layer of the UNet is 2**this_value.
            DeepFovea: n integers with n=depth specifying the number of features per layer.""")
    parser_group.add_argument('--padding', type=str, default="partial", choices=['off','zero','partial'],
                              help="UNet: The padding mode for the UNet")
    parser_group.add_argument('--batchNorm', action='store_true', help="UNet: Use batch normalization in the network")
    parser_group.add_argument('--residual', action='store_true', help="Use residual connections from input to output")
    parser_group.add_argument('--hardInput', action='store_true', help="""
        UNet:
        If true, the valid input pixels are directly copied to the output.
        This hardly enforces that the sparse input samples are preserved in the output,
        instead of relying on the network and loss function to not change them.""")
    parser_group.add_argument('--upMode', type=str, default='upsample', choices=['upconv', 'upsample'],
                              help="UNet: The upsample mode")
    parser_group.add_argument('--disableTemporal', action='store_true', help='Disables temporal consistency')
    parser_group.add_argument('--initialImage', type=str, default='zero', 
                              choices=['zero','unshaded','input'], help="""
        Specifies what should be used as the previous high res frame for the first frame of the sequence,
        when no previous image is available from the previous predition.
        Available options:
         - zero: fill everything with zeros (default)
         - unshaded: Special defaults for unshaded mode: mask=-1, normal=[0,0,1], depth=0.5, ao=1
         - input: Use the interpolated input
        """)
    parser_group.add_argument('--warpSpecialMask', action='store_true', 
                              help="if True, the mask is filled with -1 instead of 0 at the borders")
    
    # Losses
    parser_group = parser.add_argument_group("Loss")
    parser_group.add_argument('--losses', type=str, required=True, help="""
       comma-separated list of loss terms with weighting as string
       Format: <loss>:<target>:<weighting>
       with: loss in {l1, l2, tl2}
             target in {mask, normal, depth, color, ao, flow}
             weighting a positive number
        """)
    parser_group.add_argument('--lossBorderPadding', type=int, default=24, help="""
    Because flow + warping can't be accurately estimated at the borders of the image,
    the border of the input images to the loss (ground truth, low res input, prediction)
    are overwritten with zeros. The size of the border is specified by this parameter.
    Pass zero to disable this padding. Default=16 as in the TecoGAN paper.
    """)
    parser_group.add_argument('--lossAO', type=float, default=1.0, 
                              help="Strength of ambient occlusion in the loss function. Default=1")
    parser_group.add_argument('--lossAmbient', type=float, default=0.1, 
                              help="Strength of the ambient light color in the loss function's shading. Default=0.1")
    parser_group.add_argument('--lossDiffuse', type=float, default=0.1, 
                              help="Strength of the diffuse light color in the loss function's shading. Default=1.0")
    parser_group.add_argument('--lossSpecular', type=float, default=0.0, 
                              help="Strength of the specular light color in the loss function's shading. Default=0.0")

    # Training
    parser_group = parser.add_argument_group("Training")
    parser_group.add_argument('--trainBatchSize', type=int, default=4, help='training batch size')
    parser_group.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
    parser_group.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
    parser_group.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser_group.add_argument('--lrGamma', type=float, default=0.5, help='The learning rate decays every lrStep-epochs by this factor')
    parser_group.add_argument('--lrStep', type=int, default=500, help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
    parser_group.add_argument('--weightDecay', type=float, default=0, help="Weight decay (L2 penalty), if supported by the optimizer. Default=0")
    parser_group.add_argument('--optim', type=str, default="Adam", choices=['RMSprop', 'Rprop', 'Adam', 'LBFGS'], 
                        help="The optimizer")
    parser_group.add_argument('--noCuda', action='store_true', help='Disable cuda')
    parser_group.add_argument('--seed', type=int, default=123, help='random seed to use')
    parser_group.add_argument('--checkpointFreq', type=int, default=10, 
                              help='checkpoints are saved every "checkpointFreq" epoch')

    # Restore
    parser_group = parser.add_argument_group("Restore")
    parser_group.add_argument('--restore', type=int, default=-1, help="Restore training from the specified run index")
    parser_group.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
    parser_group.add_argument('--pretrained', type=str, default=None, help="Path to a pretrained generator")

    # Output
    parser_group = parser.add_argument_group("Output")
    parser_group.add_argument('--logdir', type=str, required=True, 
                        help='directory for tensorboard logs')
    parser_group.add_argument('--modeldir', type=str, required=True,
                        help='Output directory for the checkpoints')
    parser_group.add_argument('--numVisImages', type=int, default=4, 
                              help="The number of test images (see testBatchSize) that are saved for visualization, ")

    # Parse it
    opt = parser.parse_args()
    opt_dict = vars(opt)
    opt_dict['type'] = 'sparse1'

    if not opt.noCuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --noCuda")
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    device = torch.device("cpu" if opt.noCuda else "cuda")
    torch.set_num_threads(4)
    print("Device:", device)

    #########################################################
    # RESERVE OUTPUT DIRECTORY
    #########################################################

    if not os.path.exists(opt.modeldir):
        os.mkdir(opt.modeldir)
    if not os.path.exists(opt.logdir):
        os.mkdir(opt.logdir)

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
        startEpoch = 1
    else:
        # prepare restoring
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
            modelInName = os.path.join(modeldir, "model_epoch_{}.pth".format(restoreEpoch))
        startEpoch = restoreEpoch
        print('Current run: %05d'%nextRunNumber)
        runName = 'run%05d'%nextRunNumber
        logdir = os.path.join(opt.logdir, runName)
        modeldir = os.path.join(opt.modeldir, runName)

    # write settings and open tensorboard logger
    optStr = str(opt);
    print(optStr)
    with open(os.path.join(modeldir, 'info.txt'), "w") as text_file:
        text_file.write(optStr)
    writer = SummaryWriter(logdir)
    writer.add_text('info', optStr, 0)

    #########################################################
    # CREATE DATASETS
    #########################################################
    print('===> Loading datasets')
    crop_size = None
    if opt.dataset is not None:
        dataset_directories = opt.dataset.split(';')
        locator = dataset.SparseDatasetLocator(dataset_directories, opt.testFraction, opt.numberOfImages)
        training_samples = locator.get_training_samples()
        test_samples = locator.get_test_samples();
        print("Number of training sample files:", len(training_samples))
        print("Number of test sample files:", len(test_samples))
        print("Number of crops per image: ", opt.trainCropsPerImage, opt.testCropsPerImage)

        crop_size = opt.trainCropSize
        train_set = dataset.SparseDataset(
            samples=training_samples, 
            crops_per_sample=opt.trainCropsPerImage,
            crop_size=opt.trainCropSize,
            fill_ratio=opt.cropFillRate,
            buffer_size=opt.bufferSize)
        test_set = dataset.SparseDataset(
            samples=training_samples, 
            crops_per_sample=opt.testCropsPerImage,
            crop_size=opt.testCropSize,
            fill_ratio=opt.cropFillRate,
            buffer_size=opt.bufferSize)

        training_data_loader = DataLoader(
            dataset=train_set, 
            batch_size=opt.trainBatchSize,
            shuffle=False)
        testing_data_loader = DataLoader(
            dataset=test_set, 
            batch_size=opt.testBatchSize,
            shuffle=False)

    elif opt.hdf5 is not None:
        dataset_directories = opt.hdf5.split(';')
        locator = dataset.SparseDatasetLocatorHDF(
            dataset_directories, opt.testFraction, opt.smoothDense)
        crop_size = locator.crop_size()

        train_set = dataset.SparseDatasetHDF5(locator, False)
        test_set = dataset.SparseDatasetHDF5(locator, True)

        training_data_loader = DataLoader(
            dataset=train_set, 
            batch_size=opt.trainBatchSize,
            shuffle=True)
        testing_data_loader = DataLoader(
            dataset=test_set, 
            batch_size=opt.testBatchSize,
            shuffle=False)

    else:
        print("You must specify either '--dataset' or '--hdf5'!")
        exit(-1)

    """
    The DataLoader will return a tuple with:
    - sparse, then input of shape B*T*9*H*W
    - dense, the output of shape B*T*8*H*W
    for the channels, see mainSparseDatasetGenerator.py
    """

    #########################################################
    # CREATE MODEL
    #########################################################
    print('===> Building model')

    """
    interpolateInput:
    - True: input channels are directly the 9 input channels
    - False: the input channels are multiplied with abs(sparse[-1]),
             hence only contain data at the sample locations
    The absolute value of the last channel of the sparse input
     is used as mask for the partial convolutions.
    """
    def preprocessInput(sparse):
        """
        sparse: input from the dataloader of shape B*T*9*H*W
        return: (processed sparse map, mask)
        """
        mask = torch.abs(sparse[:,:,8:9,:,:])
        if not opt.interpolateInput:
            sparse = sparse * mask
        return (sparse, mask)

    if opt.externalFlow:
        input_channels = 7
        output_channels = 6
    else:
        input_channels = 9
        output_channels = 8
    input_channels_with_previous = input_channels + output_channels

    # TODO: support DeepFovea network here
    model = models.UNet(input_channels_with_previous, output_channels,
                        opt.depth, opt.filters, opt.padding,
                        opt.batchNorm, opt.residual, opt.hardInput, opt.upMode,
                        True)
    model.to(device)
    if not no_summary:
        summary(model, 
                input_size=[
                    (input_channels_with_previous, crop_size, crop_size),
                    (1, crop_size, crop_size)], 
                device=device.type)

    #single_input = torch.rand(opt.testBatchSize, input_channels_with_previous, opt.testCropSize, opt.testCropSize,
    #                          dtype=torch.float32, device=device)
    #inputMask = torch.ones(opt.testBatchSize, 1, opt.testCropSize, opt.testCropSize,
    #                          dtype=torch.float32, device=device)
    #writer.add_graph(model, (single_input, inputMask), verbose=True)
    #writer.flush()
    #writer.close()
    #exit(0)

    #########################################################
    # LOSSES
    #########################################################
    print('===> Building losses')
    criterion = losses.LossNetSparse(
        device,
        output_channels, 
        opt,
        not opt.externalFlow)
    criterion.to(device)

    #########################################################
    # OPTIMIZER
    #########################################################
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
    optimizer = createOptimizer(opt.optim, model.parameters(), 
                                lr=opt.lr, weight_decay=opt.weightDecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.lrStep, opt.lrGamma)

    #########################################################
    # PRETRAINED + RESTORE
    #########################################################
    if opt.pretrained is not None:
        checkpoint = torch.load(opt.pretrained)
        model.load_state_dict(checkpoint['model_params'])
        #only load the state dict, not the whole model
        #this asserts that the model structure is the same
        print('Using pretrained model for the generator')

    if opt.restore != -1:
        print("Restore training from run", opt.restore,"and epoch",restoreEpoch)
        checkpoint = torch.load(modelInName)
        #only load the state dict, not the whole model
        #this asserts that the model structure is the same
        model.load_state_dict(checkpoint['model_params'])
        #restore scheduler and optimizer
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

    #########################################################
    # MAIN HELPER
    #########################################################

    def train(epoch):
        epoch_loss = 0
        num_minibatch = len(train_set) // opt.trainBatchSize
        pg = ProgressBar(num_minibatch, 'Training', length=50)
        model.train()
        for iteration, batch in enumerate(training_data_loader, 0):
            pg.print_progress_bar(iteration)
            input, target = batch[0].to(device), batch[1].to(device)
            B, T, Cout, H, W = target.shape
            _, _, Cin, _, _ = input.shape
            #assert(Cout == output_channels)
            #assert(Cin == input_channels)

            input_flow = input[:,:,6:8,:,:]
            input, input_mask = preprocessInput(input)
            optimizer.zero_grad()

            previous_output = None
            loss = 0
            for j in range(T):
                # prepare input
                flow = input_flow[:,j-1,:,:,:]
                if j == 0 or opt.disableTemporal:
                    previous_input = utils.initialImage(input[:,0,0:output_channels,:,:], output_channels, 
                                                   opt.initialImage, False, 1)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                    previous_warped_loss = target[:,0,:,:,:]
                else:
                    previous_input = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow, 
                        1,
                        special_mask = opt.warpSpecialMask)
                    previous_warped_loss = previous_input
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_input),
                    dim=1)
                if opt.externalFlow:
                    # remove flow from the input and output
                    single_input = torch.cat((
                        single_input[:,:6,:,:],
                        single_input[:,8:,:,:]),
                        dim=1)
                # run generator
                prediction, _ = model(single_input, input_mask[:,j,:,:,:])
                # evaluate cost
                loss0,_ = criterion(
                    target[:,j,:output_channels,:,:], 
                    prediction, 
                    previous_warped_loss,
                    no_temporal_loss = (j==0))
                del _
                loss += loss0
                epoch_loss += loss0.item()
                # save output
                if opt.externalFlow:
                    previous_output = torch.cat([
                        torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                        utils.ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                        torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                        torch.clamp(prediction[:,5:6,:,:], 0, +1) # ao
                        ], dim=1)
                else:
                    previous_output = torch.cat([
                        torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                        utils.ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                        torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                        torch.clamp(prediction[:,5:6,:,:], 0, +1), # ao
                        torch.clamp(prediction[:,6:8,:,:], -1, +1) # flow
                        ], dim=1)

            loss.backward()
            optimizer.step()
        pg.print_progress_bar(num_minibatch)
        epoch_loss /= num_minibatch * T
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))
        writer.add_scalar('train/total_loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        scheduler.step()

    def test(epoch, save_images):
        def write_image(img, filename):
            out_img = img.cpu().detach().numpy()
            out_img *= 255.0
            out_img = out_img.clip(0, 255)
            out_img = np.uint8(out_img)
            writer.add_image(filename, out_img, epoch)

        avg_psnr = 0
        avg_losses = defaultdict(float)
        with torch.no_grad():
            num_minibatch = len(test_set) // opt.testBatchSize
            pg = ProgressBar(num_minibatch, 'Testing ', length=50)
            model.eval()
            for iteration, batch in enumerate(testing_data_loader, 0):
                pg.print_progress_bar(iteration)
                input, target = batch[0].to(device), batch[1].to(device)
                B, T, Cout, H, W = target.shape
                _, _, Cin, _, _ = input.shape
                #assert(Cout == output_channels)
                #assert(Cin == input_channels)

                input_flow = input[:,:,6:8,:,:]
                input, input_mask = preprocessInput(input)

                previous_output = None
                for j in range(T):
                    # prepare input
                    flow = input_flow[:,j-1,:,:,:]
                    if j == 0 or opt.disableTemporal:
                        previous_input = utils.initialImage(input[:,0,0:output_channels,:,:], output_channels, 
                                                       opt.initialImage, False, 1)
                        # loss takes the ground truth current image as warped previous image,
                        # to not introduce a bias and big loss for the first image
                        previous_warped_loss = target[:,0,:,:,:]
                    else:
                        previous_input = models.VideoTools.warp_upscale(
                            previous_output, 
                            flow, 
                            1,
                            special_mask = opt.warpSpecialMask)
                        previous_warped_loss = previous_input
                    single_input = torch.cat((
                            input[:,j,:,:,:],
                            previous_input),
                        dim=1)
                    if opt.externalFlow:
                        # remove flow from the input and output
                        single_input = torch.cat((
                            single_input[:,:6,:,:],
                            single_input[:,8:,:,:]),
                            dim=1)
                    # run generator
                    prediction, masks = model(single_input, input_mask[:,j,:,:,:])
                    # evaluate cost
                    loss0, loss_values = criterion(
                        target[:,j,:output_channels,:,:], 
                        prediction, 
                        previous_warped_loss,
                        no_temporal_loss = (j==0))
                    # accumulate average values
                    avg_losses['total_loss'] += loss0.item()
                    psnr = 10 * log10(1 / max(1e-10, loss_values[('mse','color')]))
                    avg_losses['psnr'] += psnr
                    for key, value in loss_values.items():
                        avg_losses[str(key)] += value
                    # save output
                    if opt.externalFlow:
                        previous_output = torch.cat([
                            torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                            utils.ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                            torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                            torch.clamp(prediction[:,5:6,:,:], 0, +1) # ao
                            ], dim=1)
                    else:
                        previous_output = torch.cat([
                            torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                            utils.ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                            torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                            torch.clamp(prediction[:,5:6,:,:], 0, +1), # ao
                            torch.clamp(prediction[:,6:8,:,:], -1, +1) # flow
                            ], dim=1)

                    # save images
                    imagesToSave = opt.numVisImages - iteration*opt.testBatchSize
                    if imagesToSave>0 and save_images:
                        # for each image in the batch
                        for b in range(min(B, imagesToSave)):
                            imgID = b + iteration * B
                            # mask
                            if j==0:
                                for layer,mask in enumerate(masks):
                                    write_image(mask[b,:,:,:], 'image%03d/debug/mask%d'%(imgID, layer))
                            if opt.disableTemporal:
                                # images, two in a row: current prediction, ground truth
                                maskPredGT = torch.cat([previous_output[b,0:1,:,:], target[b,j,0:1,:,:]], dim=2)*0.5+0.5
                                write_image(maskPredGT, 'image%03d/mask/frame%03d' % (imgID, j))
                                normalPredGT = torch.cat([previous_output[b,1:4,:,:], target[b,j,1:4,:,:]], dim=2)*0.5+0.5
                                write_image(normalPredGT, 'image%03d/normal/frame%03d' % (imgID, j))
                                depthPredGT = torch.cat([previous_output[b,4:5,:,:], target[b,j,4:5,:,:]], dim=2)
                                write_image(depthPredGT, 'image%03d/depth/frame%03d' % (imgID, j))
                                aoPredGT = torch.cat([previous_output[b,5:6,:,:], target[b,j,5:6,:,:]], dim=2)
                                write_image(aoPredGT, 'image%03d/ao/frame%03d' % (imgID, j))
                            else:
                                # images, three in a row: previous-warped, current prediction, ground truth
                                maskPredGT = torch.cat([previous_input[b,0:1,:,:], previous_output[b,0:1,:,:], target[b,j,0:1,:,:]], dim=2)*0.5+0.5
                                write_image(maskPredGT, 'image%03d/mask/frame%03d' % (imgID, j))
                                normalPredGT = torch.cat([previous_input[b,1:4,:,:], previous_output[b,1:4,:,:], target[b,j,1:4,:,:]], dim=2)*0.5+0.5
                                write_image(normalPredGT, 'image%03d/normal/frame%03d' % (imgID, j))
                                depthPredGT = torch.cat([previous_input[b,4:5,:,:], previous_output[b,4:5,:,:], target[b,j,4:5,:,:]], dim=2)
                                write_image(depthPredGT, 'image%03d/depth/frame%03d' % (imgID, j))
                                aoPredGT = torch.cat([previous_input[b,5:6,:,:], previous_output[b,5:6,:,:], target[b,j,5:6,:,:]], dim=2)
                                write_image(aoPredGT, 'image%03d/ao/frame%03d' % (imgID, j))
                            # flow
                            if opt.externalFlow:
                                flowPredGT = torch.cat([
                                    torch.cat([flow[b,:,:,:], torch.zeros_like(target[b,j,6:7,:,:])], dim=0), 
                                    torch.cat([target[b,j,6:8,:,:], torch.zeros_like(target[b,j,6:7,:,:]) ], dim=0)], dim=2)*20+0.5
                                write_image(flowPredGT, 'image%03d/flow/frame%03d' % (imgID, j))
                            else:
                                flowPredGT = torch.cat([
                                    torch.cat([previous_output[b,6:8,:,:], torch.zeros_like(previous_output[b,6:7,:,:])], dim=0), 
                                    torch.cat([target[b,j,6:8,:,:], torch.zeros_like(target[b,j,6:7,:,:]) ], dim=0)], dim=2)*20+0.5
                                write_image(flowPredGT, 'image%03d/flow/frame%03d' % (imgID, j))


            pg.print_progress_bar(num_minibatch)
        for key in avg_losses.keys():
            avg_losses[key] /= num_minibatch * T
        print("===> Avg. PSNR: {:.4f} dB".format(avg_losses['psnr']))
        print("  losses:",avg_losses)
        for key, value in avg_losses.items():
            writer.add_scalar('test/%s'%key, value, epoch)
        writer.flush()

    def checkpoint(epoch):
        model_out_path = os.path.join(modeldir, "model_epoch_{}.pth".format(epoch))
        state = {
            'epoch': epoch + 1, 
            'model_params': model.state_dict(), 
            'parameters':opt_dict,
            'optimizer':optimizer, 
            'scheduler':scheduler}
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    #########################################################
    # MAIN LOOP
    #########################################################

    print('===> Start Training')
    try:
        test(startEpoch-1, True)
        for epoch in range(startEpoch, opt.nEpochs + 1):
            train(epoch)
            save_images = (epoch < 20 or (epoch%10==0))
            test(epoch, save_images)
            if epoch%opt.checkpointFreq == 0 or epoch==1:
                checkpoint(epoch)
    except KeyboardInterrupt:
        print("Interrupted")
    writer.close()