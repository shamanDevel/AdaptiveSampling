"""
Computes statistics for dense isosurface superresolution.


"""

import math
import os
import os.path
import time
import json
import h5py
import collections

import numpy as np
import scipy.misc
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from console_progressbar import ProgressBar

import dataset.denseDatasetLoaderHDF5_v2
import models
import losses
import importance
from utils import ScreenSpaceShading, initialImage, MeanVariance, SSIM, MSSSIM, PSNR
import losses.lpips as lpips

if __name__ == "__main__":

    #########################
    # CONFIGURATION
    #########################

    if 1:
        OUTPUT_FOLDER = "../result-stats/denseIso2/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            ("Ejecta", "gt-rendering-ejecta-v2-test.hdf5", "gt-rendering-ejecta-v2-test-screen4x.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/dense-modeldir/"
        MODEL_NEAREST = "nearest"
        MODEL_BILINEAR = "bilinear"
        MODEL_BICUBIC = "bicubic"
        MODELS = [
            MODEL_NEAREST,
            MODEL_BILINEAR,
            MODEL_BICUBIC,
            "ejecta_4x_run004.pt"
            ]

        UPSCALING = 4

        LOSS_BORDER = 32
        BATCH_SIZE = 8


    #########################
    # LOADING
    #########################

    device = torch.device("cuda")

    class SimpleUpsample(nn.Module):
        def __init__(self, upscale_factor, upsample):
            super().__init__()
            self.upscale_factor = upscale_factor
            self.input_channels = 5
            self.output_channels = 6
            self.upsample = upsample

        def forward(self, inputs):
            inputs = inputs[:,0:self.input_channels,:,:]
            resized_inputs = F.interpolate(inputs, 
                                           size=[inputs.shape[2]*self.upscale_factor, 
                                                 inputs.shape[3]*self.upscale_factor], 
                                           mode=self.upsample)

            return torch.cat([
                resized_inputs,
                torch.ones(resized_inputs.shape[0],
                            self.output_channels - self.input_channels,
                            resized_inputs.shape[2],
                            resized_inputs.shape[3],
                            dtype=resized_inputs.dtype,
                            device=resized_inputs.device)],
                dim=1), None
    model_list = [None]*len(MODELS)
    for i,m in enumerate(MODELS):
        if m in [MODEL_NEAREST, MODEL_BILINEAR, MODEL_BICUBIC]:
            model_list[i] = SimpleUpsample(UPSCALING, m)
        else:
            file = os.path.join(NETWORK_DIR, m)
            extra_files = torch._C.ExtraFilesMap()
            extra_files['settings.json'] = ""
            model_list[i] = torch.jit.load(file, map_location = device, _extra_files = extra_files)
            MODELS[i] = m[:-3]

    # create shading
    shading = ScreenSpaceShading(device)
    shading.fov(30)
    shading.ambient_light_color(np.array([0.1,0.1,0.1]))
    shading.diffuse_light_color(np.array([1.0, 1.0, 1.0]))
    shading.specular_light_color(np.array([0.0, 0.0, 0.0]))
    shading.specular_exponent(16)
    shading.light_direction(np.array([0.1,0.1,1.0]))
    shading.material_color(np.array([1.0, 0.3, 0.0]))
    AMBIENT_OCCLUSION_STRENGTH = 1.0
    shading.ambient_occlusion(AMBIENT_OCCLUSION_STRENGTH)
    shading.inverse_ao = False

    #########################
    # DEFINE STATISTICS
    #########################
    ssimLoss = SSIM()
    ssimLoss.to(device)
    psnrLoss = PSNR()
    psnrLoss.to(device)
    lpipsColor = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
    MIN_FILLING = 0.05
    NUM_BINS = 200
    class Statistics:

        def __init__(self):
            self.reset()
            self.histogram_color_withAO = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_color_noAO = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_depth = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_normal = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_mask = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_ao = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_counter = 0

        def reset(self):
            self.n = 0

            self.psnr_mask = 0
            self.psnr_normal = 0
            self.psnr_depth = 0
            self.psnr_ao = 0
            self.psnr_color_withAO = 0
            self.psnr_color_noAO = 0

            self.ssim_mask = 0
            self.ssim_normal = 0
            self.ssim_depth = 0
            self.ssim_ao = 0
            self.ssim_color_withAO = 0
            self.ssim_color_noAO = 0

            self.lpips_color_withAO = 0
            self.lpips_color_noAO = 0

        def write_header(self, file):
            file.write("PSNR-mask\tPSNR-normal\tPSNR-depth\tPSNR-ao\tPSNR-color-noAO\tPSNR-color-withAO\t")
            file.write("SSIM-mask\tSSIM-normal\tSSIM-depth\tSSIM-ao\tSSIM-color-noAO\tSSIM-color-withAO\t")
            file.write("LPIPS-color-noAO\tLPIPS-color-withAO\n")

        def add_timestep_sample(self, pred_mnda, gt_mnda):
            """
            adds a timestep sample:
            pred_mnda: prediction: mask, normal, depth, AO
            gt_mnda: ground truth: mask, normal, depth, AO
            """
            #pred_mnda = pred_mnda.cpu()
            #gt_mnda = gt_mnda.cpu()

            #shading
            shading.ambient_occlusion(AMBIENT_OCCLUSION_STRENGTH)
            pred_color_withAO = shading(pred_mnda)
            gt_color_withAO = shading(gt_mnda)
            shading.ambient_occlusion(0.0)
            pred_color_noAO = shading(pred_mnda)
            gt_color_noAO = shading(gt_mnda)

            #apply border
            pred_mnda = pred_mnda[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]
            pred_color_withAO = pred_color_withAO[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]
            pred_color_noAO = pred_color_noAO[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]
            gt_mnda = gt_mnda[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]
            gt_color_withAO = gt_color_withAO[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]
            gt_color_noAO = gt_color_noAO[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]

            mask = gt_mnda[:,0:1,:,:] * 0.5 + 0.5
            self.n += 1

            # PSNR
            self.psnr_mask += psnrLoss(pred_mnda[:,0:1,:,:], gt_mnda[:,0:1,:,:]).item()
            self.psnr_normal += psnrLoss(pred_mnda[:,1:4,:,:], gt_mnda[:,1:4,:,:], mask=mask).item()
            self.psnr_depth += psnrLoss(pred_mnda[:,4:5,:,:], gt_mnda[:,4:5,:,:], mask=mask).item()
            self.psnr_ao += psnrLoss(pred_mnda[:,5:6,:,:], gt_mnda[:,5:6,:,:], mask=mask).item()
            self.psnr_color_withAO += psnrLoss(pred_color_withAO, gt_color_withAO, mask=mask).item()
            self.psnr_color_noAO += psnrLoss(pred_color_noAO, gt_color_noAO, mask=mask).item()

            # SSIM
            self.ssim_mask += ssimLoss(pred_mnda[:,0:1,:,:], gt_mnda[:,0:1,:,:]).item()
            pred_mnda = gt_mnda + mask * (pred_mnda - gt_mnda)
            self.ssim_normal += ssimLoss(pred_mnda[:,1:4,:,:], gt_mnda[:,1:4,:,:]).item()
            self.ssim_depth += ssimLoss(pred_mnda[:,4:5,:,:], gt_mnda[:,4:5,:,:]).item()
            self.ssim_ao += ssimLoss(pred_mnda[:,5:6,:,:], gt_mnda[:,5:6,:,:]).item()
            self.ssim_color_withAO += ssimLoss(pred_color_withAO, gt_color_withAO).item()
            self.ssim_color_noAO += ssimLoss(pred_color_noAO, gt_color_noAO).item()

            # Perceptual
            self.lpips_color_withAO += lpipsColor(pred_color_withAO, gt_color_withAO, normalize=True).item()
            self.lpips_color_noAO += lpipsColor(pred_color_noAO, gt_color_noAO, normalize=True).item()

            # Histogram
            self.histogram_counter += 1

            mask_diff = F.l1_loss(gt_mnda[0,0,:,:], pred_mnda[0,0,:,:], reduction='none')
            histogram,_ = np.histogram(mask_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_mask += (histogram/NUM_BINS - self.histogram_mask)/self.histogram_counter

            #normal_diff = (-F.cosine_similarity(gt_mnda[0,1:4,:,:], pred_mnda[0,1:4,:,:], dim=0)+1)/2
            normal_diff = F.l1_loss(gt_mnda[0,1:4,:,:], pred_mnda[0,1:4,:,:], reduction='none').sum(dim=0) / 6
            histogram,_ = np.histogram(normal_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_normal += (histogram/NUM_BINS - self.histogram_normal)/self.histogram_counter

            depth_diff = F.l1_loss(gt_mnda[0,4,:,:], pred_mnda[0,4,:,:], reduction='none')
            histogram,_ = np.histogram(depth_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_depth += (histogram/NUM_BINS - self.histogram_depth)/self.histogram_counter

            ao_diff = F.l1_loss(gt_mnda[0,5,:,:], pred_mnda[0,5,:,:], reduction='none')
            histogram,_ = np.histogram(ao_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_ao += (histogram/NUM_BINS - self.histogram_ao)/self.histogram_counter

            color_diff = F.l1_loss(gt_color_withAO[0,0,:,:], pred_color_withAO[0,0,:,:], reduction='none')
            histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_color_withAO += (histogram/NUM_BINS - self.histogram_color_withAO)/self.histogram_counter

            color_diff = F.l1_loss(gt_color_noAO[0,0,:,:], pred_color_noAO[0,0,:,:], reduction='none')
            histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_color_noAO += (histogram/NUM_BINS - self.histogram_color_noAO)/self.histogram_counter

        def write_sample(self, file):
            """
            All timesteps were added, write out the statistics for this sample
            and reset.
            """
            self.n = max(1, self.n)
            file.write("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n" % (
                self.psnr_mask/self.n, self.psnr_normal/self.n, self.psnr_depth/self.n, self.psnr_ao/self.n, self.psnr_color_noAO/self.n, self.psnr_color_withAO/self.n,
                self.ssim_mask/self.n, self.ssim_normal/self.n, self.ssim_depth/self.n, self.ssim_ao/self.n, self.ssim_color_noAO/self.n, self.ssim_color_withAO/self.n,
                self.lpips_color_noAO/self.n, self.lpips_color_withAO/self.n))
            #file.flush()
            self.reset()

        def write_histogram(self, file):
            """
            After every sample for the current dataset was processed, write
            a histogram of the errors in a new file
            """
            file.write("BinStart\tBinEnd\tL1ErrorMask\tL1ErrorNormal\tL1ErrorDepth\tL1ErrorAO\tL1ErrorColorWithAO\tL1ErrorColorNoAO\n")
            for i in range(NUM_BINS):
                file.write("%7.5f\t%7.5f\t%e\t%e\t%e\t%e\t%e\t%e\n" % (
                    i / NUM_BINS, (i+1) / NUM_BINS,
                    self.histogram_mask[i],
                    self.histogram_normal[i],
                    self.histogram_depth[i],
                    self.histogram_ao[i],
                    self.histogram_color_withAO[i],
                    self.histogram_color_noAO[i]
                    ))
            pass

    #########################
    # DATASET
    #########################
    class FullResDataset(torch.utils.data.Dataset):
        def __init__(self, file):
            self.file_high = h5py.File(file[0], 'r')
            self.file_low = h5py.File(file[1], 'r')
            self.dset_high = self.file_high['gt']
            self.dset_low = self.file_low['gt']
            print("Dataset shape:", self.dset_high.shape)
        def __len__(self):
            return self.dset_high.shape[0]
        def __getitem__(self, idx):
            return self.dset_high[idx,...], self.dset_low[idx,...]

    #########################
    # COMPUTE STATS for each dataset
    #########################
    for dataset_name, dataset_file_high, dataset_file_low in DATASETS:
        dataset_file_high = os.path.join(DATASET_PREFIX, dataset_file_high)
        dataset_file_low = os.path.join(DATASET_PREFIX, dataset_file_low)
        print("Compute statistics for", dataset_name)
        
        # create output folder
        output_folder = os.path.join(OUTPUT_FOLDER, dataset_name)
        os.makedirs(output_folder, exist_ok = True)

        # define statistics
        StatsCfg = collections.namedtuple(
            "StatsCfg",
            "stats network file histogram")
        statistics = []
        with open(os.path.join(output_folder, "info.txt"), 'w') as info:
            info.write("network\tfilename\thistogram\n")
            for i,m in enumerate(MODELS):
                filename = "Stats_%s.txt"%(m)
                histogram = "Histogram_%s.txt"%(m)
                info.write("%s\t%s\t%s\n"%(m, filename, histogram))
                file = open(os.path.join(output_folder, filename), "w")
                s = Statistics()
                s.write_header(file)
                statistics.append(StatsCfg(
                    stats = s,
                    network = model_list[i],
                    file = file,
                    histogram = histogram
                    ))
        print(len(statistics), " different combinations are performed per sample")

        # compute statistics
        try:
            with torch.no_grad():
                set = FullResDataset((dataset_file_high, dataset_file_low))
                data_loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)

                num_minibatch = len(data_loader)
                pg = ProgressBar(num_minibatch, 'Evaluation', length=50)
                for iteration, (batch_high, batch_low) in enumerate(data_loader, 0):
                    pg.print_progress_bar(iteration)
                    batch_high, batch_low = batch_high.to(device), batch_low.to(device)
                    B, T, C, H, W = batch_high.shape

                    # try out each combination
                    for s in statistics:
                        #print(s)

                        previous_output = None
                        for j in range(T):
                            # extract flow (always the last two channels of crop_high)
                            flow = batch_low[:,j,C-2:,:,:]

                            # prepare input
                            if j == 0:
                                previous_warped = initialImage(batch_low[:,0,:,:,:], 6, 'zero', False, UPSCALING)
                            else:
                                previous_warped = models.VideoTools.warp_upscale(
                                    previous_output, 
                                    flow, 
                                    UPSCALING,
                                    special_mask = True)
                                #previous_warped = previous_output
                            previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, UPSCALING)
                            single_input = torch.cat((
                                    batch_low[:,j,:5,:,:],
                                    previous_warped_flattened),
                                dim=1)
                            # run model
                            prediction, _ = s.network(single_input)
                            prediction[:,0:1,:,:] = torch.clamp(prediction[:,0:1,:,:], -1, +1) #mask
                            prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1) # normal
                            prediction[:,4:6,:,:] = torch.clamp(prediction[:,4:6,:,:], 0, +1) # depth+ao

                            ## DEBUG
                            #if j==0:
                            #    fig, axes = plt.subplots(ncols=3, nrows=2)
                            #    axes[0,0].imshow(batch_low[0,j,0].cpu().numpy())
                            #    axes[0,1].imshow(batch_low[0,j,1:4].permute(1,2,0).cpu().numpy())
                            #    axes[0,2].imshow(batch_low[0,j,5].cpu().numpy())
                            #    axes[0,0].set_title("Input: Mask")
                            #    axes[0,1].set_title("Input: Normal")
                            #    axes[0,2].set_title("Input: AO")
                            #    axes[1,0].imshow(prediction[0,0].cpu().numpy())
                            #    axes[1,1].imshow(prediction[0,1:4].permute(1,2,0).cpu().numpy())
                            #    axes[1,2].imshow(prediction[0,5].cpu().numpy())
                            #    axes[1,0].set_title("Reconstruction: Mask")
                            #    axes[1,1].set_title("Reconstruction: Normal")
                            #    axes[1,2].set_title("Reconstruction: AO")
                            #    plt.show()

                            # compute statistics
                            for b in range(B):
                                s.stats.add_timestep_sample(
                                    prediction[b:b+1,...],
                                    batch_high[b:b+1,j,:6,:,:])
                                s.stats.write_sample(s.file)

                            # save for next frame
                            previous_output = prediction

                    # endfor: statistic
                # endfor: batch

                pg.print_progress_bar(num_minibatch)
            # end no_grad()
        finally:
            # close files
            for s in statistics:
                with open(os.path.join(output_folder, s.histogram), "w") as histo_file:
                    s.stats.write_histogram(histo_file)
                s.file.close()

