import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import contextlib
import sys
from enum import Flag, auto
import copy
import math
import os
import imageio

from tqdm import tqdm,trange
from tqdm.contrib import DummyTqdmFile

import models
import losses
import importance
from utils import ScreenSpaceShading, initialImage, MeanVariance, SSIM, MSSSIM, PSNR, binarySearch, humanbytes
import losses.lpips as lpips
from statistics.statsLoader import StatField, HistoField
import inference

NETWORK_DTYPE = torch.float16

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

IMPORTANCE_BASELINE1 = "ibase1"
IMPORTANCE_BASELINE2 = "ibase2"
class ImportanceModel:
    def __init__(self, file, device):
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
            file = file[1] + "_importance.pt"
            extra_files = torch._C.ExtraFilesMap()
            extra_files['settings.json'] = ""
            self._net = torch.jit.load(file, map_location = device, _extra_files = extra_files)
            self._net.to(dtype=NETWORK_DTYPE)
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
        output = self._net(input.to(dtype=NETWORK_DTYPE)).to(dtype=input.dtype) # the network call
        return output

RECON_BASELINE = "rbase"
class ReconstructionModel:
    def __init__(self, file, device):
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
            file = file[1] + "_recon.pt"
            extra_files = torch._C.ExtraFilesMap()
            extra_files['settings.json'] = ""
            self._net = torch.jit.load(file, map_location = device, _extra_files = extra_files)
            self._net.to(dtype=NETWORK_DTYPE)
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
                        input = x[:, 0:8, :, :].contiguous() # mask, normal xyz, depth, ao
                        mask = x[:, 8, :, :].contiguous()
                        inpainted = torch.ops.renderer.fast_inpaint(mask, input)
                        x[:, 0:8, :, :] = inpainted
                        if self._requiresMask:
                            return self._n(x.to(dtype=NETWORK_DTYPE), mask.to(dtype=NETWORK_DTYPE)).to(dtype=x.dtype)
                        else:
                            return self._n(x.to(dtype=NETWORK_DTYPE)).to(dtype=x.dtype)
                self._net = Inpainting2(self._originalNet, requiresMask)
    def name(self):
        return self._name
    def __repr__(self):
        return self.name()

    def call(self, input, mask, prev_warped_out):
        input = torch.cat([input, prev_warped_out], dim=1)
        output = self._net(input, mask)
        return output

def setupScreenShading(settings : inference.RenderSettings):
    shading = ScreenSpaceShading(device)
    shading.fov(settings.CAM_FOV)
    shading.ambient_light_color(np.array(settings.AMBIENT_LIGHT_COLOR))
    shading.diffuse_light_color(np.array(settings.DIFFUSE_LIGHT_COLOR))
    shading.specular_light_color(np.array(settings.SPECULAR_LIGHT_COLOR))
    shading.specular_exponent(settings.SPECULAR_EXPONENT)
    shading.light_direction(np.array(settings.LIGHT_DIRECTION))
    shading.material_color(np.array(settings.MATERIAL_COLOR))
    shading.ambient_occlusion(1.0)
    shading.inverse_ao = False
    return shading

def randomPointOnSphere():
    while True:
        pos = np.random.uniform(-1.0, 1.0, size=3)
        r2 = np.dot(pos,pos)
        if r2<1e-5 and r2>1.0: continue
        return pos / np.sqrt(r2)

def benchmark(scene):
    DEBUG = False
    IMAGE_EXPORT = [(512, 512)]#[(2**9, 1024)] # screen, volume resolution

    #SETTINGS
    SCREEN_RESOLUTIONS = [2**i for i in range(6, 12)]
    print("Screen resolutions: ", SCREEN_RESOLUTIONS)
    MIN_VOLUME_RESOLUTION = 32
    NUM_SAMPLES = 50

    NETWORK_DIR = "D:/VolumeSuperResolution/adaptive-dvr-modeldir/"
    NETWORK = ("network", NETWORK_DIR+"adapDvr5-rgb-temp001-perc01-epoch500")
    VOLUME_FOLDER = "../../isosurface-super-resolution-data/volumes/cvol-filtered/"
    SAMPLING_FILE = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
    SETTINGS_FOLDER = "../network/video/"

    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda")

    # load sampling pattern
    print("load sampling patterns")
    SAMPLING_PATTERNS = ['halton', 'plastic', 'random', 'regular']
    with h5py.File(SAMPLING_FILE, 'r') as f:
        SAMPLING_PATTERN =  torch.from_numpy(f['plastic'][...]).to(device_gpu).unsqueeze(0)

    # load networks
    print("Load networks")
    importanceNetwork = ImportanceModel(NETWORK, device_gpu)
    reconNetwork = ReconstructionModel(NETWORK, device_gpu)

    # scenes
    if scene==1:
        # EJECTA 512
        VOLUME = "snapshot_272_512.cvol"
        STEPSIZE = 0.125
        POSITION_SAMPLER = lambda: (1.1**0.3) * randomPointOnSphere()
        SETTINGS = "Dvr-Ejecta-settings.json"
        UPSCALING = 8
        POSTPROCESS = importance.PostProcess(0.001, 0.05, 1, 0, 'basic')
        IMAGE_PATH = "exportDvrEjecta512_%d_%d_%d.png"
        OUTPUT_FILE = "../result-stats/DvrBenchmarkEjecta512.tsv"
    elif scene==2:
        # RM
        VOLUME = "ppmt273_1024.cvol"
        STEPSIZE = 0.25
        def rmPositionSampler():
            pos = (1.1**0.3) * randomPointOnSphere()
            pos[2] = -abs(pos[2])
            return pos
        POSITION_SAMPLER = rmPositionSampler
        SETTINGS = "Dvr-RM-settings.json"
        UPSCALING = 8
        POSTPROCESS = importance.PostProcess(0.001, 0.05, 1, 0, 'basic')
        IMAGE_PATH = "exportDvrRM1024_%d_%d_%d.png"
        OUTPUT_FILE = "../result-stats/DvrBenchmarkRM1024.tsv"
    elif scene==3:
        # RM
        VOLUME = "cleveland70.cvol"
        STEPSIZE = 0.25
        POSITION_SAMPLER = lambda: (1.1**2.4) * randomPointOnSphere()
        SETTINGS = "Dvr-Thorax-settings.json"
        UPSCALING = 8
        POSTPROCESS = importance.PostProcess(0.001, 0.05, 1, 0, 'basic')
        IMAGE_PATH = "exportDvrThorax512_%d_%d_%d.png"
        OUTPUT_FILE = "../result-stats/DvrBenchmarkThorax512.tsv"

    ###################################################
    ######## RUN BECHMARK##############################
    ###################################################

    # no gradients anywhere
    torch.set_grad_enabled(False)

    # load volume
    err = torch.ops.renderer.load_volume_from_binary(VOLUME_FOLDER+VOLUME)
    assert err==1
    resX, resY, resZ = torch.ops.renderer.get_volume_resolution()
    print("volume resolution:", resX, resY, resZ)
    minRes = max(resX, resY, resZ)
    numMipmapLevels = 0
    while minRes >= MIN_VOLUME_RESOLUTION:
        numMipmapLevels += 1
        minRes = minRes//2
    print("Num mipmap levels:", numMipmapLevels)

    # load settings
    settings = inference.RenderSettings()
    camera = inference.Camera(
        512, 
        512,
        [0,0,-1])
    with open(SETTINGS_FOLDER+SETTINGS, "r") as f:
        o = json.load(f)
        settings.from_dict(o)
        camera.from_dict(o['Camera'])
    settings.update_camera(camera)
    settings.RENDER_MODE = 2

    # run scenes
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with open(OUTPUT_FILE, "w") as stat_file:
        stat_file.write("VolumeResolution\tScreenResolution\tRenderingLowMillis\tImportanceMillis\tRenderingSamplesMillis\tReconstructionMillis\tRenderingHighMillis\tSamplePercentage\n")
        with std_out_err_redirect_tqdm() as orig_stdout:
            for mipmapLevel in range(numMipmapLevels):
                # load mipmap level
                if mipmapLevel > 0:
                    torch.ops.renderer.create_mipmap_level(mipmapLevel, "average")
                settings.MIPMAP_LEVEL = mipmapLevel
                volumeResolution = max(resX, resY, resZ) / (2**mipmapLevel)

                for resolution in SCREEN_RESOLUTIONS:
                    settings.RESOLUTION = [resolution, resolution]
                    settings.VIEWPORT = [0, 0, resolution, resolution]

                    print("volume resolution: %d, screen resolution %d"%(
                        volumeResolution, resolution))

                    # loop over sample positions
                    renderingLowMillis = 0
                    importanceMillis = 0
                    renderingSamplesMillis = 0
                    reconstructionMillis = 0
                    renderingHighMillis = 0
                    samplePercentage = 0

                    for i in trange(NUM_SAMPLES, desc='Samples', file=orig_stdout, dynamic_ncols=True, leave=True):
                        pos = list(POSITION_SAMPLER())
                        settings.CAM_ORIGIN_START = pos
                        settings.CAM_ORIGIN_END = pos
                        settings.send()

                        render_settings = settings.clone()
                        render_settings.send()

                        # high
                        torch.cuda.synchronize()
                        start.record()
                        high_res = torch.ops.renderer.render()
                        end.record()
                        torch.cuda.synchronize()
                        renderingHighMillis += start.elapsed_time(end)

                        if (resolution, volumeResolution) in IMAGE_EXPORT:
                            filename = IMAGE_PATH%(
                                resolution, volumeResolution, i)
                            image = high_res[0:3,:,:]
                            image = image.detach().cpu().numpy().transpose((1,2,0))
                            imageio.imwrite(filename, image)
                            print("Image saved to %s"%filename)

                        # low
                        settingsTmp = settings.clone()
                        settingsTmp.downsampling = UPSCALING
                        settingsTmp.send()
                        torch.cuda.synchronize()
                        start.record()
                        low_res = torch.ops.renderer.render()
                        end.record()
                        torch.cuda.synchronize()
                        renderingLowMillis += start.elapsed_time(end)

                        # prepare for importance map
                        low_res = low_res.unsqueeze(0)
                        low_res_input = low_res[:,:-2,:,:]
                        previous_input = torch.zeros(
                                1,1,
                                low_res_input.shape[2]*importanceNetwork.networkUpscaling(),
                                low_res_input.shape[3]*importanceNetwork.networkUpscaling(), 
                                dtype=low_res_input.dtype, device=low_res_input.device)

                        # compute importance map
                        torch.cuda.synchronize()
                        start.record()
                        importance_map = importanceNetwork.call(low_res_input[:,0:5,:,:], previous_input)
                        end.record()
                        torch.cuda.synchronize()
                        importanceMillis += start.elapsed_time(end)
                        if len(importance_map.shape)==3:
                            importance_map = importance_map.unsqueeze(1)
                        if DEBUG:
                            print("importance map min=%f, max=%f"%(
                                torch.min(importance_map), torch.max(importance_map)))

                        # prepare sampling
                        settings.send()
                        pattern = SAMPLING_PATTERN[
                            :, :importance_map.shape[-2], :importance_map.shape[-1]]
                        normalized_importance_map = POSTPROCESS(importance_map)[0]
                        if DEBUG:
                            print("normalized importance map min=%f, max=%f"%(
                                torch.min(normalized_importance_map), torch.max(normalized_importance_map)))
                            print("pattern min=%f, max=%f"%(
                                torch.min(pattern), torch.max(pattern)))
                        sampling_mask = normalized_importance_map > pattern
                        sample_positions = torch.nonzero(sampling_mask[0].t_())
                        sample_positions = sample_positions.to(torch.float32).transpose(0,1).contiguous()
                        samplePercentage += sample_positions.size(1) / (importance_map.shape[-2]*importance_map.shape[-1])
                        if DEBUG:
                            print("sample count: %d"%sample_positions.size(1))

                        # do the sampling
                        torch.cuda.synchronize()
                        start.record()
                        sample_data = torch.ops.renderer.render_samples(sample_positions)
                        reconstruction_input = torch.ops.renderer.scatter_samples(
                            sample_positions, sample_data, resolution, resolution,
                            [0]*10)
                        end.record()
                        torch.cuda.synchronize()
                        renderingSamplesMillis += start.elapsed_time(end)

                        # reconstruction
                        reconstruction_input = reconstruction_input[:9,:,:].unsqueeze(0)
                        previous_input = torch.zeros(
                                1,8,
                                reconstruction_input.shape[2],
                                reconstruction_input.shape[3], 
                                dtype=reconstruction_input.dtype, device=reconstruction_input.device)
                        torch.cuda.synchronize()
                        start.record()
                        reconNetwork.call(reconstruction_input, sampling_mask, previous_input)
                        end.record()
                        torch.cuda.synchronize()
                        reconstructionMillis += start.elapsed_time(end)

                    # write stats
                    renderingLowMillis /= NUM_SAMPLES
                    importanceMillis /= NUM_SAMPLES
                    renderingSamplesMillis /= NUM_SAMPLES
                    reconstructionMillis /= NUM_SAMPLES
                    renderingHighMillis /= NUM_SAMPLES
                    samplePercentage /= NUM_SAMPLES
                    stat_file.write("%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n"%(
                        volumeResolution, resolution, renderingLowMillis,
                        importanceMillis, renderingSamplesMillis,
                        reconstructionMillis, renderingHighMillis,
                        samplePercentage))
                    stat_file.flush()


if __name__ == "__main__":
    torch.ops.load_library("./Renderer.dll")
    torch.ops.renderer.initialize_renderer()
    benchmark(1)
    benchmark(2)
    benchmark(3)