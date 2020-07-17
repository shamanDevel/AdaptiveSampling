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
from statistics.statsLoader import StatField, HistoField

def run():
    torch.ops.load_library("./Renderer.dll")

    #########################
    # CONFIGURATION
    #########################

    if 0:
        OUTPUT_FOLDER = "../result-stats/adaptiveIso2/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            #("Ejecta", "gt-rendering-ejecta-v2-test.hdf5"),
            ("RM", "gt-rendering-rm-v1.hdf5"),
            #("Human", "gt-rendering-human-v1.hdf5"),
            #("Thorax", "gt-rendering-thorax-v1.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-modeldir/"
        NETWORKS = [ #suffixed with _importance.pt and _recon.pt
            #("adaptive011", "adaptive011_epoch500"), #title, file prefix
            ("adaptive019", "adaptive019_epoch470"),
            ("adaptive023", "adaptive023_epoch300")
            ]

        SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        SAMPLING_PATTERNS = ['halton', 'plastic', 'random']

        HEATMAP_MIN = [0.01, 0.05, 0.2]
        HEATMAP_MEAN = [0.05, 0.1, 0.2, 0.5]

        UPSCALING = 8 # = networkUp * postUp

        IMPORTANCE_BORDER = 8
        LOSS_BORDER = 32
        BATCH_SIZE = 8

    elif 0:
        OUTPUT_FOLDER = "../result-stats/adaptiveIsoEnhance6/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            ("Ejecta", "gt-rendering-ejecta-v2-test.hdf5"),
            #("RM", "gt-rendering-rm-v1.hdf5"),
            #("Human", "gt-rendering-human-v1.hdf5"),
            #("Thorax", "gt-rendering-thorax-v1.hdf5"),
            #("Head", "gt-rendering-head.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-modeldir/"
        NETWORKS = [ #suffixed with _importance.pt and _recon.pt
            #title, file prefix
            #("U-Net (5-4)", "sizes/size5-4_epoch500"),
            #("Enhance-Net (epoch 50)", "enhance2_imp050_epoch050"),
            #("Enhance-Net (epoch 400)", "enhance2_imp050_epoch400"),
            #("Enhance-Net (Thorax)", "enhance_imp050_Thorax_epoch200"),
            #("Enhance-Net (RM)", "enhance_imp050_RM_epoch200"),
            #("Imp100", "enhance4_imp100_epoch300"),
            #("Imp100res", "enhance4_imp100res_epoch230"),
            ("Imp100res+N", "enhance4_imp100res+N_epoch300"),
            ("Imp100+N", "enhance4_imp100+N_epoch300"),
            #("Imp100+N-res", "enhance4_imp100+N-res_epoch300"),
            #("Imp100+N-resInterp", "enhance4_imp100+N-resInterp_epoch300"),
            #("U-Net (5-4)", "size5-4_epoch500"),
            #("U-Net (5-3)", "size5-3_epoch500"),
            #("U-Net (4-4)", "size4-4_epoch500"),
            ]

        # Test if it is better to post-train with dense networks and PDE inpainting
        POSTTRAIN_NETWORK_DIR = "D:/VolumeSuperResolution/dense-modeldir/"
        POSTTRAIN_NETWORKS = [
            # title, file suffix to POSTTRAIN_NETWORK_DIR, inpainting {'fast', 'pde'}
            #("Enhance PDE (post)", "inpHv2-pde05-epoch200.pt", "pde")
            ]

        SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        SAMPLING_PATTERNS = ['plastic']

        HEATMAP_MIN = [0.002]
        HEATMAP_MEAN = [0.05] #[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        USE_BINARY_SEARCH_ON_MEAN = True

        UPSCALING = 8 # = networkUp * postUp

        IMPORTANCE_BORDER = 8
        LOSS_BORDER = 32
        BATCH_SIZE = 4

    elif 0:
        OUTPUT_FOLDER = "../result-stats/adaptiveIsoEnhance5Sampling/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            ("Ejecta", "gt-rendering-ejecta-v2-test.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-modeldir/"
        NETWORKS = [ #suffixed with _importance.pt and _recon.pt
            #title, file prefix
            ("Enhance-Net (epoch 400)", "enhance2_imp050_epoch400"),
            ]

        # Test if it is better to post-train with dense networks and PDE inpainting
        POSTTRAIN_NETWORK_DIR = "D:/VolumeSuperResolution/dense-modeldir/"
        POSTTRAIN_NETWORKS = [
            # title, file suffix to POSTTRAIN_NETWORK_DIR, inpainting {'fast', 'pde'}
            #("Enhance PDE (post)", "inpHv2-pde05-epoch200.pt", "pde")
            ]

        SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        SAMPLING_PATTERNS = ['halton', 'plastic', 'random', 'regular']
        #SAMPLING_PATTERNS = ['regular']

        HEATMAP_MIN = [0.002]
        HEATMAP_MEAN = [0.05]
        USE_BINARY_SEARCH_ON_MEAN = True

        UPSCALING = 8 # = networkUp * postUp

        IMPORTANCE_BORDER = 8
        LOSS_BORDER = 32
        BATCH_SIZE = 4

    elif 1:
        OUTPUT_FOLDER = "../result-stats/adaptiveIsoEnhance8Sampling/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            ("Ejecta", "gt-rendering-ejecta-v2-test.hdf5"),
            #("RM", "gt-rendering-rm-v1.hdf5"),
            #("Human", "gt-rendering-human-v1.hdf5"),
            #("Thorax", "gt-rendering-thorax-v1.hdf5"),
            #("Head", "gt-rendering-head.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-modeldir/"
        NETWORKS = [ #suffixed with _importance.pt and _recon.pt
            #title, file prefix
            ("regular", "enhance7_regular_epoch190"),
            ("random", "enhance7_random_epoch190"),
            ("halton", "enhance7_halton_epoch190"),
            ("plastic", "enhance7_plastic_epoch190"),
            ]

        # Test if it is better to post-train with dense networks and PDE inpainting
        POSTTRAIN_NETWORK_DIR = "D:/VolumeSuperResolution/dense-modeldir/"
        POSTTRAIN_NETWORKS = [
            # title, file suffix to POSTTRAIN_NETWORK_DIR, inpainting {'fast', 'pde'}
            #("Enhance PDE (post)", "inpHv2-pde05-epoch200.pt", "pde")
            ]

        SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        SAMPLING_PATTERNS = ['regular', 'random', 'halton', 'plastic']

        HEATMAP_MIN = [0.002]
        HEATMAP_MEAN = [0.05] #[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        USE_BINARY_SEARCH_ON_MEAN = True

        UPSCALING = 8 # = networkUp * postUp

        IMPORTANCE_BORDER = 8
        LOSS_BORDER = 32
        BATCH_SIZE = 4

    elif 0:
        OUTPUT_FOLDER = "../result-stats/adaptiveIsoImp/"
        DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
        DATASETS = [
            ("Ejecta", "gt-rendering-ejecta-v2-test.hdf5"),
            #("RM", "gt-rendering-rm-v1.hdf5"),
            #("Human", "gt-rendering-human-v1.hdf5"),
            #("Thorax", "gt-rendering-thorax-v1.hdf5"),
            ]

        NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-modeldir/imp/"
        NETWORKS = [ #suffixed with _importance.pt and _recon.pt
            #("adaptive011", "adaptive011_epoch500"), #title, file prefix
            ("imp005", "imp005_epoch500"),
            ("imp010", "imp010_epoch500"),
            ("imp020", "imp020_epoch500"),
            ("imp050", "imp050_epoch500"),
            ]

        SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
        SAMPLING_PATTERNS = ['halton']

        HEATMAP_MIN = [0.002]
        HEATMAP_MEAN = [0.005, 0.01, 0.02, 0.05, 0.1]
        USE_BINARY_SEARCH_ON_MEAN = True

        UPSCALING = 8 # = networkUp * postUp

        IMPORTANCE_BORDER = 8
        LOSS_BORDER = 32
        BATCH_SIZE = 16

    #########################
    # LOADING
    #########################

    device = torch.device("cuda")

    # Load Networks
    IMPORTANCE_BASELINE1 = "ibase1"
    IMPORTANCE_BASELINE2 = "ibase2"
    IMPORTANCE_BASELINE3 = "ibase3"
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
                self._net = importance.GradientImportanceMap(1, (1,1),(2,1),(3,1))
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

    class LuminanceImportanceModel:
        def __init__(self):
            self.disableTemporal = True
        def setTestFile(self, filename):
            importance_file = filename[:-5] + "-luminanceImportance.hdf5"
            if os.path.exists(importance_file):
                self._exist = True
                self._file = h5py.File(importance_file, 'r')
                self._dset = self._file['importance']
            else:
                self._exist = False
                self._file = None
                self._dset = None
        def isAvailable(self):
            return self._exist
        def setIndices(self, indices : torch.Tensor):
            assert len(indices.shape) == 1
            self._indices = list(indices.cpu().numpy())
        def setTime(self, time):
            self._time = time
        def networkUpscaling(self):
            return UPSCALING
        def name(self):
            return "luminance-contrast"
        def __repr__(self):
            return self.name()
        def call(self, input, prev_warped_out):
            B, C, H, W = input.shape
            if not self._exist:
                return torch.ones(B, 1, H, W, dtype=input.dtype, device=input.device)
            outputs = []
            for idx in self._indices:
                outputs.append(torch.from_numpy(self._dset[idx, self._time, ...]).to(device=input.device))
            return torch.stack(outputs, dim=0)

    importanceBaseline1 = ImportanceModel(IMPORTANCE_BASELINE1)
    importanceBaseline2 = ImportanceModel(IMPORTANCE_BASELINE2)
    importanceBaselineLuminance = LuminanceImportanceModel()
    importanceModels = [ImportanceModel(f) for f in NETWORKS]

    # load reconstruction networks
    print("load reconstruction networks")
    class ReconstructionModel:
        def __init__(self, file):
            if file == RECON_BASELINE:
                class Inpainting(nn.Module):
                    def forward(self, x, mask):
                        input = x[:, 0:6, :, :].contiguous() # mask, normal xyz, depth, ao
                        mask = x[:, 6, :, :].contiguous()
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
                            input = x[:, 0:6, :, :].contiguous() # mask, normal xyz, depth, ao
                            mask = x[:, 6, :, :].contiguous()
                            inpainted = torch.ops.renderer.fast_inpaint(mask, input)
                            x[:, 0:6, :, :] = inpainted
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

        def __init__(self, name : str, model_path : str, inpainting : str):
            assert inpainting=='fast' or inpainting=='pde', "inpainting must be either 'fast' or 'pde', but got %s"%inpainting
            self._inpainting = inpainting

            self._name = name
            file = os.path.join(POSTTRAIN_NETWORK_DIR, model_path)
            extra_files = torch._C.ExtraFilesMap()
            extra_files['settings.json'] = ""
            self._net = torch.jit.load(file, map_location = device, _extra_files = extra_files)
            self._settings = json.loads(extra_files['settings.json'])
            assert self._settings.get('upscale_factor', None)==1, "selected file is not a 1x SRNet"
            self.disableTemporal = False

        def name(self):
            return self._name
        def __repr__(self):
            return self.name()

        def call(self, input, prev_warped_out):
            # no sampling and no AO
            input_no_sampling = input[:, 0:5, :, :].contiguous() # mask, normal xyz, depth
            sampling_mask = input[:, 6, :, :].contiguous()
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
            input = torch.cat([inpainted, prev_warped_out], dim=1)
            output = self._net(input)
            if isinstance(output, tuple):
                output = output[0]
            return output

    reconBaseline = ReconstructionModel(RECON_BASELINE)
    reconModels = [ReconstructionModel(f) for f in NETWORKS]
    reconPostModels = [ReconstructionModelPostTrain(name, file, inpainting)
                       for (name, file, inpainting) in POSTTRAIN_NETWORKS]
    allReconModels = reconModels + reconPostModels

    NETWORK_COMBINATIONS = \
        [(importanceBaseline1, reconBaseline), (importanceBaseline2, reconBaseline)] + \
        [(importanceBaselineLuminance, reconBaseline)] + \
        [(importanceBaseline1, reconNet) for reconNet in allReconModels] + \
        [(importanceBaseline2, reconNet) for reconNet in allReconModels] + \
        [(importanceBaselineLuminance, reconNet) for reconNet in allReconModels] + \
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
    shading.ambient_occlusion(1.0)
    shading.inverse_ao = False

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
            self.histogram_color_withAO = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_color_noAO = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_depth = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_normal = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_mask = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_ao = np.zeros(NUM_BINS, dtype=np.float64)
            self.histogram_counter = 0

        def create_datasets(self, 
                            hdf5_file : h5py.File, 
                            stats_name : str, histo_name : str, 
                            num_samples : int,
                            extra_info : dict):

            self.expected_num_samples = num_samples
            stats_shape = (num_samples, len(list(StatField)))
            self.stats_file = hdf5_file.require_dataset(
                stats_name, stats_shape, dtype='f', exact=True)
            self.stats_file.attrs['NumFields'] = len(list(StatField))
            for field in list(StatField):
                self.stats_file.attrs['Field%d'%field.value] = field.name
            for key, value in extra_info.items():
                self.stats_file.attrs[key] = value
            self.stats_index = 0

            histo_shape = (NUM_BINS, len(list(HistoField)))
            self.histo_file = hdf5_file.require_dataset(
                histo_name, histo_shape, dtype='f', exact=True)
            self.histo_file.attrs['NumFields'] = len(list(HistoField))
            for field in list(HistoField):
                self.histo_file.attrs['Field%d'%field.value] = field.name
            for key, value in extra_info.items():
                self.histo_file.attrs[key] = value

        def add_timestep_sample(self, pred_mnda, gt_mnda, sampling_mask):
            """
            adds a timestep sample:
            pred_mnda: prediction: mask, normal, depth, AO
            gt_mnda: ground truth: mask, normal, depth, AO
            """
            B = pred_mnda.shape[0]

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

            # PSNR
            psnr_mask = psnrLoss(pred_mnda[:,0:1,:,:], gt_mnda[:,0:1,:,:]).cpu().numpy()
            psnr_normal = psnrLoss(pred_mnda[:,1:4,:,:], gt_mnda[:,1:4,:,:], mask=mask).cpu().numpy()
            psnr_depth = psnrLoss(pred_mnda[:,4:5,:,:], gt_mnda[:,4:5,:,:], mask=mask).cpu().numpy()
            psnr_ao = psnrLoss(pred_mnda[:,5:6,:,:], gt_mnda[:,5:6,:,:], mask=mask).cpu().numpy()
            psnr_color_withAO = psnrLoss(pred_color_withAO, gt_color_withAO, mask=mask).cpu().numpy()
            psnr_color_noAO = psnrLoss(pred_color_noAO, gt_color_noAO, mask=mask).cpu().numpy()

            # SSIM
            ssim_mask = ssimLoss(pred_mnda[:,0:1,:,:], gt_mnda[:,0:1,:,:]).cpu().numpy()
            pred_mnda = gt_mnda + mask * (pred_mnda - gt_mnda)
            ssim_normal = ssimLoss(pred_mnda[:,1:4,:,:], gt_mnda[:,1:4,:,:]).cpu().numpy()
            ssim_depth = ssimLoss(pred_mnda[:,4:5,:,:], gt_mnda[:,4:5,:,:]).cpu().numpy()
            ssim_ao = ssimLoss(pred_mnda[:,5:6,:,:], gt_mnda[:,5:6,:,:]).cpu().numpy()
            ssim_color_withAO = ssimLoss(pred_color_withAO, gt_color_withAO).cpu().numpy()
            ssim_color_noAO = ssimLoss(pred_color_noAO, gt_color_noAO).cpu().numpy()

            # Perceptual
            lpips_color_withAO = torch.cat([lpipsColor(pred_color_withAO[b], gt_color_withAO[b], normalize=True) for b in range(B)], dim=0).cpu().numpy()
            lpips_color_noAO = torch.cat([lpipsColor(pred_color_noAO[b], gt_color_noAO[b], normalize=True) for b in range(B)], dim=0).cpu().numpy()

            # Samples
            samples = torch.mean(sampling_mask, dim=(1,2,3)).cpu().numpy()

            # Write samples to file
            for b in range(B):
                assert self.stats_index < self.expected_num_samples, "Adding more samples than specified"
                self.stats_file[self.stats_index, :] = np.array([
                    psnr_mask[b], psnr_normal[b], psnr_depth[b], psnr_ao[b], psnr_color_noAO[b], psnr_color_withAO[b],
                    ssim_mask[b], ssim_normal[b], ssim_depth[b], ssim_ao[b], ssim_color_noAO[b], ssim_color_withAO[b],
                    lpips_color_noAO[b], lpips_color_withAO[b],
                    samples[b]], dtype='f')
                self.stats_index += 1

            # Histogram
            self.histogram_counter += 1

            mask_diff = F.l1_loss(gt_mnda[:,0,:,:], pred_mnda[:,0,:,:], reduction='none')
            histogram,_ = np.histogram(mask_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_mask += (histogram/(NUM_BINS*B) - self.histogram_mask)/self.histogram_counter

            #normal_diff = (-F.cosine_similarity(gt_mnda[0,1:4,:,:], pred_mnda[0,1:4,:,:], dim=0)+1)/2
            normal_diff = F.l1_loss(gt_mnda[:,1:4,:,:], pred_mnda[:,1:4,:,:], reduction='none').sum(dim=0) / 6
            histogram,_ = np.histogram(normal_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_normal += (histogram/(NUM_BINS*B) - self.histogram_normal)/self.histogram_counter

            depth_diff = F.l1_loss(gt_mnda[:,4,:,:], pred_mnda[:,4,:,:], reduction='none')
            histogram,_ = np.histogram(depth_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_depth += (histogram/(NUM_BINS*B) - self.histogram_depth)/self.histogram_counter

            ao_diff = F.l1_loss(gt_mnda[:,5,:,:], pred_mnda[:,5,:,:], reduction='none')
            histogram,_ = np.histogram(ao_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_ao += (histogram/(NUM_BINS*B) - self.histogram_ao)/self.histogram_counter

            color_diff = F.l1_loss(gt_color_withAO[:,0,:,:], pred_color_withAO[:,0,:,:], reduction='none')
            histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_color_withAO += (histogram/(NUM_BINS*B) - self.histogram_color_withAO)/self.histogram_counter

            color_diff = F.l1_loss(gt_color_noAO[:,0,:,:], pred_color_noAO[:,0,:,:], reduction='none')
            histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
            self.histogram_color_noAO += (histogram/(NUM_BINS*B) - self.histogram_color_noAO)/self.histogram_counter

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
                    self.histogram_mask[i],
                    self.histogram_normal[i],
                    self.histogram_depth[i],
                    self.histogram_ao[i],
                    self.histogram_color_withAO[i],
                    self.histogram_color_noAO[i]
                    ])

    #########################
    # DATASET
    #########################
    class FullResDataset(torch.utils.data.Dataset):
        def __init__(self, file):
            self.hdf5_file = h5py.File(file, 'r')
            self.dset = self.hdf5_file['gt']
            print("Dataset shape:", self.dset.shape)
        def __len__(self):
            return self.dset.shape[0]
        def num_timesteps(self):
            return self.dset.shape[1]
        def __getitem__(self, idx):
            return (self.dset[idx,...], np.array(idx))

    #########################
    # COMPUTE STATS for each dataset
    #########################
    for dataset_name, dataset_file in DATASETS:
        dataset_file = os.path.join(DATASET_PREFIX, dataset_file)
        print("Compute statistics for", dataset_name)
        
        # init luminance importance map
        importanceBaselineLuminance.setTestFile(dataset_file)
        if importanceBaselineLuminance.isAvailable():
            print("Luminance-contrast importance map is available")

        # create output file
        os.makedirs(OUTPUT_FOLDER, exist_ok = True)
        output_file = os.path.join(OUTPUT_FOLDER, dataset_name+'.hdf5')
        print("Save to", output_file)
        with h5py.File(output_file, 'a') as output_hdf5_file:

            # load dataset
            set = FullResDataset(dataset_file)
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
                            inet.name(), rnet.name(), heatmin*100, heatmean*100, pattern)
                        histo_filename = "Histogram_%s_%s_%03d_%03d_%s"%(
                            inet.name(), rnet.name(), heatmin*100, heatmean*100, pattern)
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
                    for iteration, (batch, batch_indices) in enumerate(data_loader, 0):
                        pg.print_progress_bar(iteration)
                        batch = batch.to(device)
                        importanceBaselineLuminance.setIndices(batch_indices)
                        B, T, C, H, W = batch.shape

                        # try out each combination
                        for s in statistics:
                            #print(s)
                            # get input to evaluation
                            importanceNetUpscale = s.importance.networkUpscaling()
                            importancePostUpscale = UPSCALING // importanceNetUpscale
                            crop_low = torch.nn.functional.interpolate(
                                batch.reshape(B*T, C, H, W), scale_factor=1/UPSCALING,
                                mode='area').reshape(B, T, C, H//UPSCALING, W//UPSCALING)
                            pattern = sampling_pattern[s.pattern][:H, :W]
                            crop_high = batch

                            # loop over timesteps
                            pattern = pattern.unsqueeze(0).unsqueeze(0)
                            previous_importance = None
                            previous_output = None
                            reconstructions = []
                            for j in range(T):
                                importanceBaselineLuminance.setTime(j)
                                # extract flow (always the last two channels of crop_high)
                                flow = crop_high[:,j,C-2:,:,:]

                                # compute importance map
                                importance_input = crop_low[:,j,:5,:,:]
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
                                    sample_mask * crop_high[:,j,0:5,:,:], # mask, normal x, normal y, normal z, depth
                                    sample_mask * torch.ones(B,1,H,W, dtype=crop_high.dtype, device=crop_high.device), # ao
                                    sample_mask), # sample mask
                                    dim = 1)

                                # warp previous output
                                if j==0 or s.recon.disableTemporal:
                                    previous_input = torch.zeros(B,6,H,W, dtype=crop_high.dtype, device=crop_high.device)
                                else:
                                    previous_input = models.VideoTools.warp_upscale(
                                        previous_output,
                                        flow,
                                        1, False)

                                # run reconstruction network
                                reconstruction = s.recon.call(reconstruction_input, sample_mask, previous_input)

                                # clamp
                                reconstruction_clamped = torch.cat([
                                    torch.clamp(reconstruction[:,0:1,:,:], -1, +1), # mask
                                    ScreenSpaceShading.normalize(reconstruction[:,1:4,:,:], dim=1),
                                    torch.clamp(reconstruction[:,4:5,:,:], 0, +1), # depth
                                    torch.clamp(reconstruction[:,5:6,:,:], 0, +1) # ao
                                    ], dim=1)
                                reconstructions.append(reconstruction_clamped)

                                # save for next frame
                                previous_output = reconstruction_clamped

                            #endfor: timesteps

                            # compute statistics
                            reconstructions = torch.cat(reconstructions, dim=0)
                            crops_high = torch.cat([crop_high[:,j,:6,:,:] for j in range(T)], dim=0)
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