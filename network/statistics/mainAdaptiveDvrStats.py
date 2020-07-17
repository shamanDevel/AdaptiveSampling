"""
Computes statistics for adaptive isosurface rendering.

Test cases:
1. Different network combinations with baseline, statistics of PSNR+SSIM+REC distribution
2. Influence of minImportance, meanImportance as a graph of PSNR+SSIM+REC
3. PSNR+SSIM+REC for different sample pattern

"""

import math
import os
import os.path
import time
import json
import h5py
import collections
import sys

import numpy as np
import scipy.misc
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from console_progressbar import ProgressBar

import dataset.adaptiveDatasetLoader
import models
import losses
import importance
from utils import ScreenSpaceShading, initialImage, MeanVariance, SSIM, MSSSIM, PSNR, binarySearch
import losses.lpips as lpips
from statistics.statsLoader import StatFieldDvr, HistoFieldDvr

def run():
    torch.ops.load_library("./Renderer.dll")

    #########################
    # CONFIGURATION
    #########################

    if 1:
        OUTPUT_FOLDER = "../result-stats/adaptiveDvr3/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            ("Ejecta", "gt-dvr-ejecta6-test.hdf5", "gt-dvr-ejecta6-test-screen8x.hdf5"),
            ("RM", "gt-dvr-rm1-test.hdf5", "gt-dvr-rm1-test-screen8x.hdf5"),
            ("Thorax", "gt-dvr-thorax2-test.hdf5", "gt-dvr-thorax2-test-screen8x.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-dvr-modeldir/"
        NETWORKS = [ #suffixed with _importance.pt and _recon.pt
            #title, file prefix
            #("v5-temp001", "adapDvr5-rgb-temp001-epoch300"),
            #("v5-temp010", "adapDvr5-rgb-temp010-epoch300"),
            #("v5-temp100", "adapDvr5-rgb-temp100-epoch300"),
            #("v5-temp001-perc", "adapDvr5-rgb-temp001-perc01-epoch300"),
            ("v5-perc01+bn", "adapDvr5-rgb-perc01-bn-epoch500"),
            ("v5-perc01-bn", "adapDvr5-rgb-temp001-perc01-epoch500")
            ]

        # Test if it is better to post-train with dense networks and PDE inpainting
        POSTTRAIN_NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-dvr-modeldir/"
        POSTTRAIN_NETWORKS = [
            # title, file suffix to POSTTRAIN_NETWORK_DIR, inpainting {'fast', 'pde'}
            #("Enhance PDE (post)", "inpHv2-pde05-epoch200.pt", "pde")
            ("v6pr2-noTemp", "ejecta6pr2-plastic05-lpips-noTempCon-epoch500_recon.pt", "fast", False),
            ("v6pr2-tl2-100", "ejecta6pr2-plastic05-lpips-tl2-100-epoch500_recon.pt", "fast", True)
            ]

        SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        SAMPLING_PATTERNS = ['plastic']

        HEATMAP_MIN = [0.002]
        HEATMAP_MEAN = [0.02, 0.05, 0.1, 0.2] #[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        USE_BINARY_SEARCH_ON_MEAN = True

        UPSCALING = 8 # = networkUp * postUp

        IMPORTANCE_BORDER = 8
        LOSS_BORDER = 32
        BATCH_SIZE = 1#2

    #########################
    # LOADING
    #########################

    device = torch.device("cuda")

    # Load Networks
    IMPORTANCE_BASELINE1 = "ibase1"
    IMPORTANCE_BASELINE2 = "ibase2"
    RECON_BASELINE = "rbase"

    # load importance model
    print("load importance networks")
    class ImportanceModel:
        def __init__(self, file):
            if file == IMPORTANCE_BASELINE1:
                self._net = importance.UniformImportanceMap(1, 0.5)
                self._upscaling = 1
                self._name = "constant"
                self.disableTemporal = True
                self._requiresPrevious = False
            elif file == IMPORTANCE_BASELINE2:
                self._net = importance.GradientImportanceMap(1, (0,1),(1,1),(2,1))
                self._upscaling = 1
                self._name = "curvature"
                self.disableTemporal = True
                self._requiresPrevious = False
            else:
                self._name = file[0]
                file = os.path.join(NETWORK_DIR, file[1] + "_importance.pt")
                extra_files = torch._C.ExtraFilesMap()
                extra_files['settings.json'] = ""
                self._net = torch.jit.load(file, map_location = device, _extra_files = extra_files)
                settings = json.loads(extra_files['settings.json'])
                self._upscaling = settings['networkUpscale']
                self._requiresPrevious = settings.get("requiresPrevious", False)
                self.disableTemporal = settings.get("disableTemporal", True)

        def networkUpscaling(self):
            return self._upscaling
        def name(self):
            return self._name
        def __repr__(self):
            return self.name()

        def call(self, input, prev_warped_out):
            if self._requiresPrevious:
                input = torch.cat([
                    input,
                    models.VideoTools.flatten_high(prev_warped_out, self._upscaling)
                    ], dim=1)
            input = F.pad(input, 
                          [IMPORTANCE_BORDER]*4,
                          'constant', 0)
            output = self._net(input) # the network call
            output = F.pad(output, 
                           [-IMPORTANCE_BORDER*self._upscaling]*4, 
                           'constant', 0)
            return output
    importanceBaseline1 = ImportanceModel(IMPORTANCE_BASELINE1)
    importanceBaseline2 = ImportanceModel(IMPORTANCE_BASELINE2)
    importanceModels = [ImportanceModel(f) for f in NETWORKS]

    # load reconstruction networks
    print("load reconstruction networks")
    class ReconstructionModel:
        def __init__(self, file):
            if file == RECON_BASELINE:
                class Inpainting(nn.Module):
                    def forward(self, x, mask):
                        input = x[:, 0:4, :, :].contiguous() # rgba, don't use normal xyz, depth
                        mask = x[:, 8, :, :].contiguous()
                        return torch.ops.renderer.fast_inpaint(mask, input)
                self._net = Inpainting()
                self._upscaling = 1
                self._name = "inpainting"
                self.disableTemporal = True
            else:
                self._name = file[0]
                file = os.path.join(NETWORK_DIR, file[1] + "_recon.pt")
                extra_files = torch._C.ExtraFilesMap()
                extra_files['settings.json'] = ""
                self._net = torch.jit.load(file, map_location = device, _extra_files = extra_files)
                self._settings = json.loads(extra_files['settings.json'])
                self.disableTemporal = False
                requiresMask = self._settings.get('expectMask', False)
                if self._settings.get("interpolateInput", False):
                    self._originalNet = self._net
                    class Inpainting2(nn.Module):
                        def __init__(self, orignalNet, requiresMask):
                            super().__init__()
                            self._n = orignalNet
                            self._requiresMask = requiresMask
                        def forward(self, x, mask):
                            input = x[:, 0:8, :, :].contiguous() # rgba, normal xyz, depth
                            mask = x[:, 8, :, :].contiguous()
                            inpainted = torch.ops.renderer.fast_inpaint(mask, input)
                            x[:, 0:8, :, :] = inpainted
                            if self._requiresMask:
                                return self._n(x, mask)
                            else:
                                return self._n(x)
                    self._net = Inpainting2(self._originalNet, requiresMask)
        def name(self):
            return self._name
        def __repr__(self):
            return self.name()

        def call(self, input, mask, prev_warped_out):
            input = torch.cat([input, prev_warped_out], dim=1)
            output = self._net(input, mask)
            return output

    class ReconstructionModelPostTrain:
        """
        Reconstruction model that are trained as dense reconstruction networks
        after the adaptive training.
        They don't recive the sampling mask as input, but can start with PDE-based inpainting
        """

        def __init__(self, name : str, model_path : str, inpainting : str, has_temporal : bool):
            assert inpainting=='fast' or inpainting=='pde', "inpainting must be either 'fast' or 'pde', but got %s"%inpainting
            self._inpainting = inpainting

            self._name = name
            file = os.path.join(POSTTRAIN_NETWORK_DIR, model_path)
            extra_files = torch._C.ExtraFilesMap()
            extra_files['settings.json'] = ""
            self._net = torch.jit.load(file, map_location = device, _extra_files = extra_files)
            self._settings = json.loads(extra_files['settings.json'])
            assert self._settings.get('upscale_factor', None)==1, "selected file is not a 1x SRNet"
            self.disableTemporal = not has_temporal

        def name(self):
            return self._name
        def __repr__(self):
            return self.name()

        def call(self, input, mask, prev_warped_out):
            # no sampling and no AO
            input_no_sampling = input[:, 0:8, :, :].contiguous() # rgba, normal xyz, depth
            sampling_mask = mask[:, 0, :, :].contiguous()
            # perform inpainting
            if self._inpainting == 'pde':
                inpainted = torch.ops.renderer.pde_inpaint(
                    sampling_mask, input_no_sampling,
                    200, 1e-4, 5, 2, # m0, epsilon, m1, m2
                    0, # mc -> multigrid recursion count. =0 disables the multigrid hierarchy
                    9, 0) # ms, m3
            else:
                inpainted = torch.ops.renderer.fast_inpaint(
                    sampling_mask, input_no_sampling)
            # run network
            if self.disableTemporal:
                prev_warped_out = torch.zeros_like(prev_warped_out)
            input = torch.cat([inpainted, prev_warped_out], dim=1)
            output = self._net(input)
            if isinstance(output, tuple):
                output = output[0]
            return output

    reconBaseline = ReconstructionModel(RECON_BASELINE)
    reconModels = [ReconstructionModel(f) for f in NETWORKS]
    reconPostModels = [ReconstructionModelPostTrain(name, file, inpainting, has_temporal)
                       for (name, file, inpainting, has_temporal) in POSTTRAIN_NETWORKS]
    allReconModels = reconModels + reconPostModels

    NETWORK_COMBINATIONS = \
        [(importanceBaseline1, reconBaseline), (importanceBaseline2, reconBaseline)] + \
        [(importanceBaseline1, reconNet) for reconNet in allReconModels] + \
        [(importanceBaseline2, reconNet) for reconNet in allReconModels] + \
        [(importanceNet, reconBaseline) for importanceNet in importanceModels] + \
        list(zip(importanceModels, reconModels)) + \
        [(importanceNet, reconPostModel) for importanceNet in importanceModels for reconPostModel in reconPostModels]
    #NETWORK_COMBINATIONS = list(zip(importanceModels, reconModels))
    print("Network combinations:")
    for (i,r) in NETWORK_COMBINATIONS:
        print("  %s - %s"%(i.name(), r.name()))

    # load sampling patterns
    print("load sampling patterns")
    with h5py.File(SAMPLING_FILE, 'r') as f:
        sampling_pattern = dict([(name, torch.from_numpy(f[name][...]).to(device)) \
            for name in SAMPLING_PATTERNS])

    #heatmap
    HEATMAP_CFG = [(min, mean) for min in HEATMAP_MIN for mean in HEATMAP_MEAN if min<mean]
    print("heatmap configs:", HEATMAP_CFG)

    #########################
    # DEFINE STATISTICS
    #########################
    ssimLoss = SSIM(size_average=False)
    ssimLoss.to(device)
    psnrLoss = PSNR()
    psnrLoss.to(device)
    lpipsColor = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
    MIN_FILLING = 0.05
    NUM_BINS = 200    

    class Statistics:

        def __init__(self):
            self.histogram_color = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_alpha = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_counter = 0

        def create_datasets(self, 
                            hdf5_file : h5py.File, 
                            stats_name : str, histo_name : str, 
                            num_samples : int,
                            extra_info : dict):

            self.stats_name = stats_name
            self.expected_num_samples = num_samples
            stats_shape = (num_samples, len(list(StatFieldDvr)))
            self.stats_file = hdf5_file.require_dataset(
                stats_name, stats_shape, dtype='f', exact=True)
            self.stats_file.attrs['NumFields'] = len(list(StatFieldDvr))
            for field in list(StatFieldDvr):
                self.stats_file.attrs['Field%d'%field.value] = field.name
            for key, value in extra_info.items():
                self.stats_file.attrs[key] = value
            self.stats_index = 0

            histo_shape = (NUM_BINS, len(list(HistoFieldDvr)))
            self.histo_file = hdf5_file.require_dataset(
                histo_name, histo_shape, dtype='f', exact=True)
            self.histo_file.attrs['NumFields'] = len(list(HistoFieldDvr))
            for field in list(HistoFieldDvr):
                self.histo_file.attrs['Field%d'%field.value] = field.name
            for key, value in extra_info.items():
                self.histo_file.attrs[key] = value

        def add_timestep_sample(self, pred_rgba, gt_rgba, sampling_mask):
            """
            adds a timestep sample:
            pred_rgba: prediction rgba
            gt_rgba: ground truth rgba
            """
            B = pred_rgba.shape[0]

            #apply border
            pred_rgba = pred_rgba[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]
            gt_rgba = gt_rgba[:,:,LOSS_BORDER:-LOSS_BORDER,LOSS_BORDER:-LOSS_BORDER]

            # PSNR
            psnr_color = psnrLoss(pred_rgba[:,0:3,:,:], gt_rgba[:,0:3,:,:]).cpu().numpy()
            psnr_alpha = psnrLoss(pred_rgba[:,3:4,:,:], gt_rgba[:,3:4,:,:]).cpu().numpy()

            # SSIM
            ssim_color = ssimLoss(pred_rgba[:,0:3,:,:], gt_rgba[:,0:3,:,:]).cpu().numpy()
            ssim_alpha = ssimLoss(pred_rgba[:,3:4,:,:], gt_rgba[:,3:4,:,:]).cpu().numpy()

            # Perceptual
            lpips_color = torch.cat([ \
                lpipsColor(pred_rgba[b, 0:3, :, :], gt_rgba[b, 0:3, :, :], normalize=True) \
                    for b in range(B)], dim=0).cpu().numpy()

            # Samples
            samples = torch.mean(sampling_mask, dim=(1,2,3)).cpu().numpy()

            # Write samples to file
            for b in range(B):
                assert self.stats_index < self.expected_num_samples, "Adding more samples than specified"
                self.stats_file[self.stats_index, :] = np.array([
                    psnr_color[b], psnr_alpha[b],
                    ssim_color[b], ssim_alpha[b],
                    lpips_color[b], samples[b]], dtype='f')
                self.stats_index += 1

            # Histogram
            self.histogram_counter += 1

            color_diff = F.l1_loss(gt_rgba[:,0:3,:,:], pred_rgba[:,0:3,:,:], reduction='none').sum(dim=0) / 6
            histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_color += (histogram/(NUM_BINS*B) - self.histogram_color)/self.histogram_counter

            alpha_diff = F.l1_loss(gt_rgba[:,3,:,:], pred_rgba[:,3,:,:], reduction='none')
            histogram,_ = np.histogram(alpha_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_alpha += (histogram/(NUM_BINS*B) - self.histogram_alpha)/self.histogram_counter

        def close_stats_file(self):
            self.stats_file.attrs['NumEntries'] = self.stats_index

        def write_histogram(self):
            """
            After every sample for the current dataset was processed, write
            a histogram of the errors in a new file
            """
            for i in range(NUM_BINS):
                self.histo_file[i,:] = np.array([
                    i / NUM_BINS, (i+1) / NUM_BINS,
                    self.histogram_color[i],
                    self.histogram_alpha[i]
                    ])

    #########################
    # DATASET
    #########################
    class FullResDataset(torch.utils.data.Dataset):
        def __init__(self, file_high, file_low):
            self.hdf5_file_high = h5py.File(file_high, 'r')
            self.dset_high = self.hdf5_file_high['gt']
            self.hdf5_file_low = h5py.File(file_low, 'r')
            self.dset_low = self.hdf5_file_low['gt']
            print("Dataset shape:", self.dset_high.shape)
        def __len__(self):
            return self.dset_high.shape[0]
        def num_timesteps(self):
            return self.dset_high.shape[1]
        def __getitem__(self, idx):
            return self.dset_high[idx,...], self.dset_low[idx,...]

    #########################
    # COMPUTE STATS for each dataset
    #########################
    for dataset_name, dataset_file_high, dataset_file_low in DATASETS:
        dataset_file_high = os.path.join(DATASET_PREFIX, dataset_file_high)
        dataset_file_low = os.path.join(DATASET_PREFIX, dataset_file_low)
        print("Compute statistics for", dataset_name)
        
        # create output file
        os.makedirs(OUTPUT_FOLDER, exist_ok = True)
        output_file = os.path.join(OUTPUT_FOLDER, dataset_name+'.hdf5')
        print("Save to", output_file)
        with h5py.File(output_file, 'a') as output_hdf5_file:

            # load dataset
            set = FullResDataset(dataset_file_high, dataset_file_low)
            data_loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)

            # define statistics
            StatsCfg = collections.namedtuple(
                "StatsCfg",
                "stats importance recon heatmin heatmean pattern")
            statistics = []
            for (inet,rnet) in NETWORK_COMBINATIONS:
                for (heatmin, heatmean) in HEATMAP_CFG:
                    for pattern in SAMPLING_PATTERNS:
                        stats_info = {
                            'importance' : inet.name(),
                            'reconstruction' : rnet.name(),
                            'heatmin' : heatmin,
                            'heatmean' : heatmean,
                            'pattern' : pattern
                            }
                        stats_filename = "Stats_%s_%s_%03d_%03d_%s"%(
                            inet.name(), rnet.name(), heatmin*1000, heatmean*1000, pattern)
                        histo_filename = "Histogram_%s_%s_%03d_%03d_%s"%(
                            inet.name(), rnet.name(), heatmin*1000, heatmean*1000, pattern)
                        s = Statistics()
                        s.create_datasets(
                            output_hdf5_file,
                            stats_filename, histo_filename,
                            len(set) * set.num_timesteps(),
                            stats_info)
                        statistics.append(StatsCfg(
                            stats = s,
                            importance = inet,
                            recon = rnet,
                            heatmin = heatmin,
                            heatmean = heatmean,
                            pattern = pattern
                            ))
            print(len(statistics), " different combinations are performed per sample")

            # compute statistics
            try:
                with torch.no_grad():
                    num_minibatch = len(data_loader)
                    pg = ProgressBar(num_minibatch, 'Evaluation', length=50)
                    for iteration, (crop_high, crop_low) in enumerate(data_loader, 0):
                        pg.print_progress_bar(iteration)
                        crop_high = crop_high.to(device)
                        crop_low = crop_low.to(device)
                        B, T, C, H, W = crop_high.shape
                        _, _, _, Hlow, Wlow = crop_low.shape
                        assert Hlow*UPSCALING == H

                        # try out each combination
                        for s in statistics:
                            #print(s)
                            # get input to evaluation
                            importanceNetUpscale = s.importance.networkUpscaling()
                            importancePostUpscale = UPSCALING // importanceNetUpscale
                            pattern = sampling_pattern[s.pattern][:H, :W]

                            # loop over timesteps
                            pattern = pattern.unsqueeze(0).unsqueeze(0)
                            previous_importance = None
                            previous_output = None
                            reconstructions = []
                            for j in range(T):
                                # extract flow (always the last two channels of crop_high)
                                flow = crop_high[:,j,C-2:,:,:]

                                # compute importance map
                                importance_input = crop_low[:,j,:8,:,:]
                                if j==0 or s.importance.disableTemporal:
                                    previous_input = torch.zeros(
                                        B,1,
                                        importance_input.shape[2]*importanceNetUpscale,
                                        importance_input.shape[3]*importanceNetUpscale, 
                                        dtype=crop_high.dtype, device=crop_high.device)
                                else:
                                    flow_low = F.interpolate(flow, scale_factor = 1/importancePostUpscale)
                                    previous_input = models.VideoTools.warp_upscale(
                                        previous_importance,
                                        flow_low,
                                        1, 
                                        False)
                                importance_map = s.importance.call(importance_input, previous_input)
                                if len(importance_map.shape)==3:
                                    importance_map = importance_map.unsqueeze(1)
                                previous_importance = importance_map

                                target_mean = s.heatmean
                                if USE_BINARY_SEARCH_ON_MEAN:
                                    # For regular sampling, the normalization does not work properly,
                                    # use binary search on the heatmean instead
                                    def f(x):
                                        postprocess = importance.PostProcess(
                                            s.heatmin, x, importancePostUpscale, 
                                            LOSS_BORDER//importancePostUpscale,
                                            'basic')
                                        importance_map2 = postprocess(importance_map)[0].unsqueeze(1)
                                        sampling_mask = (importance_map2 >= pattern).to(dtype=importance_map.dtype)
                                        samples = torch.mean(sampling_mask).item()
                                        return samples
                                    target_mean = binarySearch(f, s.heatmean, s.heatmean, 10, 0, 1)
                                    #print("Binary search for #samples, mean start={}, result={} with samples={}, original={}".
                                    #      format(s.heatmean, s.heatmean, f(target_mean), f(s.heatmean)))

                                # normalize and upscale importance map
                                postprocess = importance.PostProcess(
                                    s.heatmin, target_mean, importancePostUpscale, 
                                    LOSS_BORDER//importancePostUpscale,
                                    'basic')
                                importance_map = postprocess(importance_map)[0].unsqueeze(1)
                                #print("mean:", torch.mean(importance_map).item())

                                # create samples
                                sample_mask = (importance_map >= pattern).to(dtype=importance_map.dtype)

                                reconstruction_input = torch.cat((
                                    sample_mask * crop_high[:,j,0:8,:,:], # rgba, normal xyz, depth
                                    sample_mask), # sample mask
                                    dim = 1)

                                # warp previous output
                                if j==0 or s.recon.disableTemporal:
                                    previous_input = torch.zeros(B,4,H,W, dtype=crop_high.dtype, device=crop_high.device)
                                else:
                                    previous_input = models.VideoTools.warp_upscale(
                                        previous_output,
                                        flow,
                                        1, False)

                                # run reconstruction network
                                reconstruction = s.recon.call(reconstruction_input, sample_mask, previous_input)

                                # clamp
                                reconstruction_clamped = torch.clamp(reconstruction, 0, 1)
                                reconstructions.append(reconstruction_clamped)

                                ## test
                                #if j==0:
                                #    plt.figure()
                                #    plt.imshow(reconstruction_clamped[0,0:3,:,:].cpu().numpy().transpose((1,2,0)))
                                #    plt.title(s.stats.stats_name)
                                #    plt.show()

                                # save for next frame
                                previous_output = reconstruction_clamped

                            #endfor: timesteps

                            # compute statistics
                            reconstructions = torch.cat(reconstructions, dim=0)
                            crops_high = torch.cat([crop_high[:,j,:8,:,:] for j in range(T)], dim=0)
                            sample_masks = torch.cat([sample_mask]*T, dim=0)
                            s.stats.add_timestep_sample(
                                reconstructions,
                                crops_high,
                                sample_masks)

                        # endfor: statistic
                    # endfor: batch

                    pg.print_progress_bar(num_minibatch)
                # end no_grad()
            finally:
                # close files
                for s in statistics:
                    s.stats.write_histogram()
                    s.stats.close_stats_file()
        # end with: hdf5 file
    # end for: loop over datasets

if __name__ == "__main__":
    run()

    #import pprofile
    #prof = pprofile.Profile()
    #with prof:
    #    run()
    ##prof.print_stats()
    #with open("profile.txt", 'w') as f:
    #    prof.annotate(f)

    #import cProfile
    #cProfile.run('run()')