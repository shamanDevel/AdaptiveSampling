"""
Generates random views of volumes at different isovalues
and from different camera angles.
The images are created in full resolution for the target images.
The input images can then be derived from those images by taking only specific samples.

Let W, H be the width and height of the images that are rendered.
Typically, W=1024, H=1024

Let T be the number of timesteps per sequence.
Typically, the sequences are 10 frames long

For each view, it the following numpy arrays, packed in HDF5-files:
 - dense, the target of the network of shape T*C*H*W
   with C=8 and the channels are defined as:
    - mask (+1=intersection, -1=no intersection)
    - normalX [-1,+1],
    - normalY [-1,+1],
    - normalZ [-1,+1],
    - depth [0,1],
    - ao [0,1] where 1 means no occlusion, 0 full occlusion
    - flowX [-1,+1] inpainted flow in normalized coordinates
    - flowY [-1,+1] inpainted flow in normalized coordinates

The optical flow is filled with the fast inpainting method.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from console_progressbar import ProgressBar
import json
import cv2 as cv
import h5py
import glob
from contextlib import ExitStack
from collections import namedtuple
import kornia.color as kc

from utils import FixedDict
from utils import humanbytes
from utils import load_tf_v2

class GroundTruthDatasetGenerator:

    STEPSIZE = "stepsize"
    INTERPOLATION = "interpolation"
    RESOLUTION = "resolution"
    TIMESTEPS = "timesteps"
    CAM_UP = "orientation"
    CAM_ORIGIN_START = "origin_start"
    CAM_LOOKAT_START = "lookat_start"
    CAM_ORIGIN_END = "origin_end"
    CAM_LOOKAT_END = "lookat_end"
    MIPMAP_LEVEL = "mipmap_level"

    RENDER_MODE = "render_mode"

    ISOVALUE_START = "isovalue_start"
    ISOVALUE_END = "isovalue_end"
    AO_SAMPLES = "aosamples"
    AO_RADIUS = "aoradius"

    OPACITY_SCALING = "opacity_scaling"
    DENSITY_AXIS_OPACITY = "density_axis_opacity"
    OPACITY_AXIS = "opacity_axis"
    DENSITY_AXIS_COLOR = "density_axis_color"
    COLOR_AXIS = "color_axis"
    MIN_DENSITY = "min_density"
    MAX_DENSITY = "max_density"

    DVR_USE_SHADING = "dvrUseShading"
    AMBIENT_LIGHT_COLOR = "ambientLightColor"
    DIFFUSE_LIGHT_COLOR = "diffuseLightColor"
    SPECULAR_LIGHT_COLOR = "specularLightColor"
    SPECULAR_EXPONENT = "specularExponent"
    MATERIAL_COLOR = "materialColor"
    LIGHT_DIRECTION = "lightDirection"

    VALUE_SCALING = "valueScaling"

    def getSettingsDict(self):
        d = {GroundTruthDatasetGenerator.STEPSIZE : 0.1,
             GroundTruthDatasetGenerator.INTERPOLATION : 1,
             GroundTruthDatasetGenerator.RESOLUTION : [1024, 1024],
             GroundTruthDatasetGenerator.TIMESTEPS : 10,
             GroundTruthDatasetGenerator.CAM_UP : [0,-1,0],
             GroundTruthDatasetGenerator.CAM_ORIGIN_START : [0.5,0,0],
             GroundTruthDatasetGenerator.CAM_LOOKAT_START : [0,0,0],
             GroundTruthDatasetGenerator.CAM_ORIGIN_END : [0.5,0,0],
             GroundTruthDatasetGenerator.CAM_LOOKAT_END : [0,0,0],
             GroundTruthDatasetGenerator.MIPMAP_LEVEL : 0,

             GroundTruthDatasetGenerator.RENDER_MODE : 0, #0=iso, 2=dvr

             GroundTruthDatasetGenerator.ISOVALUE_START : 0.3,
             GroundTruthDatasetGenerator.ISOVALUE_END : 0.3,
             GroundTruthDatasetGenerator.AO_SAMPLES : 256,
             GroundTruthDatasetGenerator.AO_RADIUS : 0.2,

             GroundTruthDatasetGenerator.OPACITY_SCALING: 40.0,
             GroundTruthDatasetGenerator.DENSITY_AXIS_OPACITY: [],
             GroundTruthDatasetGenerator.OPACITY_AXIS: [],
             GroundTruthDatasetGenerator.DENSITY_AXIS_COLOR: [],
             GroundTruthDatasetGenerator.COLOR_AXIS: [],
             GroundTruthDatasetGenerator.MIN_DENSITY: 0.0,
             GroundTruthDatasetGenerator.MAX_DENSITY: 1.0,

             GroundTruthDatasetGenerator.DVR_USE_SHADING : 1,
             GroundTruthDatasetGenerator.AMBIENT_LIGHT_COLOR : [0.1,0.1,0.1],
             GroundTruthDatasetGenerator.DIFFUSE_LIGHT_COLOR : [0.9,0.9,0.9],
             GroundTruthDatasetGenerator.SPECULAR_LIGHT_COLOR : [0.1,0.1,0.1],
             GroundTruthDatasetGenerator.SPECULAR_EXPONENT : 16,
             GroundTruthDatasetGenerator.MATERIAL_COLOR : [1.0,1.0,1.0],
             GroundTruthDatasetGenerator.LIGHT_DIRECTION : [0,0,1],
             GroundTruthDatasetGenerator.VALUE_SCALING : 1.0,
             }
        return FixedDict(d)

    def loadLibrary(self, path):
        torch.ops.load_library(path)
        #torch.ops.renderer.create_offscreen_context()
        torch.ops.renderer.initialize_renderer()

    def loadVolume(self, path):
        torch.ops.renderer.close_volume()
        _, extension = os.path.splitext(path)
        if extension=='.xyz':
            return torch.ops.renderer.load_volume_from_xyz(path)==1
        elif extension=='.raw':
            return torch.ops.renderer.load_volume_from_raw(path)==1
        elif extension=='.cvol':
            return torch.ops.renderer.load_volume_from_binary(path)==1
        else:
            assert False, "unknown extension " + extension

    @staticmethod
    def randomPointOnSphere():
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
        vec[2] = - abs(vec[2])
        return vec;

    @staticmethod
    def randomFloat(min, max):
        return min + np.random.random() * (max-min)

    def render(self, settingsDict, ignoreMinSamples = False):
        """
        Renders the sequence.
        settingsDict: dictionary with the settings
        outputPathPrefix: the prefix for the output files,
          they are appended by "dense.npy" or "sparse.npy"
        returns: the numpy array or None if unsuccessfull (no samples)
        """
        
        resolution = np.array(settingsDict[GroundTruthDatasetGenerator.RESOLUTION])
        W, H = settingsDict[GroundTruthDatasetGenerator.RESOLUTION]
        T = settingsDict[GroundTruthDatasetGenerator.TIMESTEPS]

        # set rendering parameters
        set_params = torch.ops.renderer.set_renderer_parameter
        VIEWPORT_OFFSET = 100
        set_params("resolution", "%d,%d"%(W, H))
        set_params("renderMode", "%d"%settingsDict[GroundTruthDatasetGenerator.RENDER_MODE])
        set_params("aosamples", "%d"%settingsDict[GroundTruthDatasetGenerator.AO_SAMPLES])
        set_params("aoradius", "%f"%settingsDict[GroundTruthDatasetGenerator.AO_RADIUS])
        set_params("cameraUp", "%f,%f,%f"%tuple(settingsDict[GroundTruthDatasetGenerator.CAM_UP]))
        set_params("stepsize", "%f"%settingsDict[GroundTruthDatasetGenerator.STEPSIZE])
        set_params("interpolation", "%d"%settingsDict[GroundTruthDatasetGenerator.INTERPOLATION])
        set_params("mipmapLevel", "%d"%settingsDict[GroundTruthDatasetGenerator.MIPMAP_LEVEL])

        set_params("opacityScaling", "%f"%settingsDict[GroundTruthDatasetGenerator.OPACITY_SCALING])
        set_params("densityAxisOpacity", ",".join(settingsDict[GroundTruthDatasetGenerator.DENSITY_AXIS_OPACITY]))
        set_params("opacityAxis", ",".join(settingsDict[GroundTruthDatasetGenerator.OPACITY_AXIS]))
        set_params("densityAxisColor", ",".join(settingsDict[GroundTruthDatasetGenerator.DENSITY_AXIS_COLOR]))
        set_params("colorAxis", ",".join(settingsDict[GroundTruthDatasetGenerator.COLOR_AXIS]))
        set_params("minDensity", "%f"%settingsDict[GroundTruthDatasetGenerator.MIN_DENSITY])
        set_params("maxDensity", "%f"%settingsDict[GroundTruthDatasetGenerator.MAX_DENSITY])

        set_params("dvrUseShading", "%d"%settingsDict[GroundTruthDatasetGenerator.DVR_USE_SHADING])
        set_params("ambientLightColor", "%f,%f,%f"%tuple(settingsDict[GroundTruthDatasetGenerator.AMBIENT_LIGHT_COLOR]))
        set_params("diffuseLightColor", "%f,%f,%f"%tuple(settingsDict[GroundTruthDatasetGenerator.DIFFUSE_LIGHT_COLOR]))
        set_params("specularLightColor", "%f,%f,%f"%tuple(settingsDict[GroundTruthDatasetGenerator.SPECULAR_LIGHT_COLOR]))
        set_params("specularExponent", "%d"%settingsDict[GroundTruthDatasetGenerator.SPECULAR_EXPONENT])
        set_params("materialColor", "%f,%f,%f"%tuple(settingsDict[GroundTruthDatasetGenerator.MATERIAL_COLOR]))
        set_params("lightDirection", "%f,%f,%f"%tuple(settingsDict[GroundTruthDatasetGenerator.LIGHT_DIRECTION]))

        def lerp(a, b, t):
            return a * (1-t) + b * t
        camOriginStart = np.array(settingsDict[GroundTruthDatasetGenerator.CAM_ORIGIN_START])
        camOriginEnd   = np.array(settingsDict[GroundTruthDatasetGenerator.CAM_ORIGIN_END])
        camTargetStart = np.array(settingsDict[GroundTruthDatasetGenerator.CAM_LOOKAT_START])
        camTargetEnd   = np.array(settingsDict[GroundTruthDatasetGenerator.CAM_LOOKAT_END])
        isoStart = settingsDict[GroundTruthDatasetGenerator.ISOVALUE_START]
        isoEnd = settingsDict[GroundTruthDatasetGenerator.ISOVALUE_END]

        valueScaling = settingsDict[GroundTruthDatasetGenerator.VALUE_SCALING]

        cudaDevice = torch.device("cuda")

        # srender dense images
        # we start at frame -1 to get the flow right but discard it's result
        set_params("viewport", "0,0,-1,-1")
        output = []
        for frame in range(-1, T):
            if frame==-1:
                set_params("aosamples", "0")
            else:
                set_params("aosamples", "%d"%settingsDict[GroundTruthDatasetGenerator.AO_SAMPLES])
            t = frame / (T-1)

            set_params("cameraOrigin", "%f,%f,%f"%tuple(lerp(camOriginStart, camOriginEnd, t).tolist()))
            set_params("cameraLookAt", "%f,%f,%f"%tuple(lerp(camTargetStart, camTargetEnd, t).tolist()))
            set_params("isovalue", "%f"%lerp(isoStart, isoEnd, t))

            denseOutput = torch.ops.renderer.render()
            if frame==-1:
                # check if enough samples
                if settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 0: #ISO
                    numHitSamples = torch.sum(denseOutput[0,:,:]).item()
                elif settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 2: #DVR
                    numHitSamples = torch.sum(denseOutput[3,:,:] > 0.0).item()
                    
                if numHitSamples <= 0.1 * W * H and not ignoreMinSamples:
                    #print("too few samples")
                    return None

                if settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 2:
                    if valueScaling is None:
                        # normalize color by value in HSV
                        color_rgb = denseOutput[0:3,:,:]
                        color_hsv = kc.rgb_to_hsv(color_rgb)
                        max_value = torch.max(color_hsv[2,:,:]).item()
                        #print("Normalize by HSV value, current:", max_value)
                        valueScaling = 1 / max_value

                continue
                
            if settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 0: #ISO
                denseOutput[0] = denseOutput[0] * 2 - 1 # convert mask from [0,1] to [-1,1]
            elif settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 2: #DVR
                # normalize hsv value (brightness)
                color_rgb = denseOutput[0:3,:,:]
                color_hsv = kc.rgb_to_hsv(color_rgb)
                color_hsv[2,:,:] *= valueScaling
                color_rgb = kc.hsv_to_rgb(color_hsv)
                denseOutput[0:3,:,:] = color_rgb

            # inpaint flow
            if settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 0: #ISO
                denseOutput[6:8,:,:] = torch.ops.renderer.fast_inpaint(
                    denseOutput[0:1,:,:], denseOutput[6:8,:,:].unsqueeze(0));
            elif settingsDict[GroundTruthDatasetGenerator.RENDER_MODE] == 2: #DVR
                denseOutput[8:10,:,:] = torch.ops.renderer.fast_inpaint_fractional(
                    denseOutput[3:4,:,:], denseOutput[8:10,:,:].unsqueeze(0));

            denseOutput = denseOutput.cpu().numpy()
            output.append(denseOutput)

            #plotting = True
            #if plotting:
            #    plt.ion()
            #
            #    ax = plt.gca()
            #    ax.set_title("Normal %d/%d" % (frame+1, T))
            #    ax.imshow((denseOutput[1:4,:,:]*0.5+0.5).transpose(1,2,0))
            #    #ax.imshow(denseOutput[5,:,:])
            #    plt.draw()
            #    print("Press any key to continue...")
            #    import msvcrt
            #    import time
            #    while True:
            #        plt.pause(0.01)
            #        if msvcrt.kbhit():
            #            break
            #
            #    ax = plt.gca()
            #    ax.set_title("Flow %d/%d" % (frame+1, T))
            #    ax.imshow((np.concatenate(
            #        [denseOutput[6:8,:,:], np.zeros((1,H, W), dtype=denseOutput.dtype)], axis=0)
            #         *10+0.5).transpose(1,2,0))
            #    #ax.imshow(denseOutput[5,:,:])
            #    plt.draw()
            #    print("Press any key to continue...")
            #    import msvcrt
            #    import time
            #    while True:
            #        plt.pause(0.01)
            #        if msvcrt.kbhit():
            #            break

        # write result
        output_array_dense = np.stack(output, axis=0)
        settingsDict[GroundTruthDatasetGenerator.VALUE_SCALING] = valueScaling
        return output_array_dense

def generateRandomSamplesIso(generator : GroundTruthDatasetGenerator, 
                          descriptor_file : str,
                          output_file : str,
                          numImages = 20, # then number of images to create per dataset
                          aoSamples = 256, # ao samples
                          inputMipmapLevel = 0, # mipmap level, see Voklume::getLevel()
                          inputMipmapFilter = "average", # mipmap level, see Voklume::getLevel()
                          save_config_file = None, # if a string, save camera positions and so on to that csv file
                          restore_config_file = None, # if a string, restore the camera positions from that csv file
                          downsampling_factor = 1):
    SIZE = 512 / downsampling_factor
    T = 10

    if save_config_file is not None and restore_config_file is not None:
        raise ValueError("either 'save_config_file' or 'restore_config_file' can be non None, not both")

    with ExitStack() as stack:
        f = stack.enter_context(h5py.File(output_file, "w"))
        if save_config_file is not None:
            config_file = open(save_config_file, "w")
            config_file.write(
                "isoStart\tisoEnd\tupX\tupY\tupZ\t" +
                "originStartX\toriginStartY\toriginStartZ\t" +
                "originEndX\toriginEndY\toriginEndZ\t" +
                "lookAtStartX\tlookAtStartY\tlookAtStartZ\t" + 
                "lookAtEndX\tlookAtEndY\tlookAtEndZ\n")
        if restore_config_file is not None:
            config = np.loadtxt(restore_config_file, skiprows=1)
            if config.shape[1] != 17:
                raise ValueError("Config file must contain 17 columns, but there are only", config.shape[1])
            print(config.shape[0], "sample configurations restored")

        if dset_name in f.keys():
            del f[dset_name]
        dset = f.create_dataset(
            "gt",
            (1, T, 8, SIZE, SIZE),
            dtype=np.float32,
            chunks = (1, 1, 8, SIZE, SIZE),
            maxshape = (None, T, 8, SIZE, SIZE))
        dset.attrs["Mode"] = "IsoUnshaded"

        settings = generator.getSettingsDict()

        # common configuration
        settings[GroundTruthDatasetGenerator.STEPSIZE] = 0.1
        settings[GroundTruthDatasetGenerator.INTERPOLATION] = 1
        settings[GroundTruthDatasetGenerator.AO_SAMPLES] = aoSamples
        settings[GroundTruthDatasetGenerator.AO_RADIUS] = 0.3
        settings[GroundTruthDatasetGenerator.TIMESTEPS] = T
        settings[GroundTruthDatasetGenerator.RESOLUTION] = [SIZE, SIZE]
        settings[GroundTruthDatasetGenerator.MIPMAP_LEVEL] = inputMipmapLevel
        settings[GroundTruthDatasetGenerator.RENDER_MODE] = 0
        maxDist = 0.3

        propOnlyCamera = 0.8
        propOnlyIso = 1.0

        #list all datasets
        dataset_info = np.genfromtxt(descriptor_file, skip_header=1, dtype=None)
        num_files = dataset_info.shape[0]
        print('Datasets:')
        for j in range(num_files):
            name = str(dataset_info[j][0].decode('ascii'))
            min_iso = float(dataset_info[j][1])
            max_iso = float(dataset_info[j][2])
            print(name,"  iso=[%f,%f]"%(min_iso, max_iso))

        expected_filesize = SIZE * SIZE * 8 * T * num_files * numImages * 4
        print("Shape: B=%d, T=%d, C=%d, H=%d, W=%d"%(num_files*numImages, T, 8, SIZE, SIZE))
        print("Expected filesize:", humanbytes(expected_filesize))

        sample_index = 0
        for j in range(num_files):
            name = str(dataset_info[j][0].decode('ascii'))
            min_iso = float(dataset_info[j][1])
            max_iso = float(dataset_info[j][2])
            inputFile = os.path.abspath(os.path.join(os.path.dirname(descriptor_file), name))
            print("Process", inputFile)
            if not generator.loadVolume(inputFile):
                print("Unable to load volume")
                continue
            if inputMipmapLevel>0:
                print("Create mipmap level")
                torch.ops.renderer.create_mipmap_level(inputMipmapLevel, inputMipmapFilter)

            pg = ProgressBar(numImages, 'Render', length=50)
            i = 0
            while i < numImages:
                pg.print_progress_bar(i)

                if restore_config_file is None:
                    originStart = GroundTruthDatasetGenerator.randomPointOnSphere() * GroundTruthDatasetGenerator.randomFloat(0.6, 1.0) + np.array([0,0,-0.07])
                    lookAtStart = GroundTruthDatasetGenerator.randomPointOnSphere() * 0.1  + np.array([0,0,-0.07])
                    while True:
                        originEnd = GroundTruthDatasetGenerator.randomPointOnSphere() * GroundTruthDatasetGenerator.randomFloat(0.6, 1.0) + np.array([0,0,-0.07])
                        if np.linalg.norm(originEnd - originStart) < maxDist:
                            break
                    lookAtEnd = GroundTruthDatasetGenerator.randomPointOnSphere() * 0.1 + np.array([0,0,-0.07])
                    up = GroundTruthDatasetGenerator.randomPointOnSphere()
                    isoStart = GroundTruthDatasetGenerator.randomFloat(min_iso, max_iso)
                    isoEnd = GroundTruthDatasetGenerator.randomFloat(min_iso, max_iso)

                    prop = np.random.random()
                    if prop <= propOnlyCamera:
                        isoEnd = isoStart
                    else: # prop <= propOnlyIso
                        originEnd = originStart
                        lookAtEnd = lookAtStart

                    settings[GroundTruthDatasetGenerator.ISOVALUE_START] = isoStart
                    settings[GroundTruthDatasetGenerator.ISOVALUE_END] = isoEnd
                    settings[GroundTruthDatasetGenerator.CAM_UP] = list(up)
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_START] = list(originStart)
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_END] = list(originEnd)
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_START] = list(lookAtStart)
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_END] = list(lookAtEnd)

                    output = generator.render(settings)
                    if output is not None:
                        dset.resize(sample_index+1, axis=0)
                        dset[sample_index,...] = output
                        sample_index += 1
                        i += 1

                        if save_config_file is not None:
                            config_file.write(
                                "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(
                                    isoStart, isoEnd, up[0], up[1], up[2],
                                    originStart[0], originStart[1], originStart[2],
                                    originEnd[0], originEnd[1], originEnd[2],
                                    lookAtStart[0], lookAtStart[1], lookAtStart[2],
                                    lookAtEnd[0], lookAtEnd[1], lookAtEnd[2]
                                    ))

                else:
                    # use configuration from the settings file
                    cfg = config[sample_index,:]
                    settings[GroundTruthDatasetGenerator.ISOVALUE_START] = cfg[0]
                    settings[GroundTruthDatasetGenerator.ISOVALUE_END] = cfg[1]
                    settings[GroundTruthDatasetGenerator.CAM_UP] = [cfg[2], cfg[3], cfg[4]]
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_START] = [cfg[5], cfg[6], cfg[7]]
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_END] = [cfg[8], cfg[9], cfg[10]]
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_START] = [cfg[11], cfg[12], cfg[13]]
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_END] = [cfg[14], cfg[15], cfg[16]]

                    output = generator.render(settings, ignoreMinSamples=True)
                    assert output is not None
                    dset.resize(sample_index+1, axis=0)
                    dset[sample_index,...] = output
                    sample_index += 1
                    i += 1

            pg.print_progress_bar(numImages)

def generateRandomSamplesDVR(generator : GroundTruthDatasetGenerator, 
                          descriptor_file : str,
                          tfDirectory : str, # folder with transfer functions
                          output_file : str,
                          numImages = 20, # then number of images to create per dataset
                          inputMipmapLevel = 0, # mipmap level, see Voklume::getLevel()
                          inputMipmapFilter = "average", # mipmap level, see Voklume::getLevel()
                          save_config_file = None, # if a string, save camera positions and so on to that csv file
                          restore_config_file = None, # if a string, restore the camera positions from that csv file
                          downsampling_factor = 1,
                          step_size = 0.1,
                          opacity_scaling = [40, 80],  #min+max
                          camera_distance = [0.5, 1.0],
                          resolution = 512,
                          dset_name = 'gt',
                          origin_mapping = lambda x : x,
                          lookAt_mapping = lambda x : x,
                          up_mapping = lambda x : x):
    SIZE = resolution / downsampling_factor
    T = 10

    if save_config_file is not None and restore_config_file is not None:
        raise ValueError("either 'save_config_file' or 'restore_config_file' can be non None, not both")

    with ExitStack() as stack:
        f = stack.enter_context(h5py.File(output_file, "a"))
        if save_config_file is not None:
            config_file = open(save_config_file, "w")
            config_file.write(
                "valid\tupX\tupY\tupZ\t" +
                "originStartX\toriginStartY\toriginStartZ\t" +
                "originEndX\toriginEndY\toriginEndZ\t" +
                "lookAtStartX\tlookAtStartY\tlookAtStartZ\t" + 
                "lookAtEndX\tlookAtEndY\tlookAtEndZ\t" + 
                "useShading\tspecularExponent\t" + 
                "lightDirX\tlightDirY\tlightDirZ\t" + 
                "tfIndex\topacityScaling\tvalueScaling\n")
        if restore_config_file is not None:
            config = np.loadtxt(restore_config_file, skiprows=1)
            if config.shape[1] != 24:
                raise ValueError("Config file must contain 21 columns, but there are only", config.shape[1])
            print(config.shape[0], "sample configurations restored")

        if dset_name in f.keys():
            del f[dset_name]
        dset = f.create_dataset(
            dset_name,
            (1, T, 10, SIZE, SIZE),
            dtype=np.float32,
            chunks = (1, 1, 10, SIZE, SIZE),
            maxshape = (None, T, 10, SIZE, SIZE))
        dset.attrs["Mode"] = "DVR"

        settings = generator.getSettingsDict()

        # common configuration
        settings[GroundTruthDatasetGenerator.STEPSIZE] = step_size
        settings[GroundTruthDatasetGenerator.INTERPOLATION] = 1
        settings[GroundTruthDatasetGenerator.TIMESTEPS] = T
        settings[GroundTruthDatasetGenerator.RESOLUTION] = [SIZE, SIZE]
        settings[GroundTruthDatasetGenerator.MIPMAP_LEVEL] = inputMipmapLevel
        settings[GroundTruthDatasetGenerator.RENDER_MODE] = 2
        maxDist = 0.3

        propOnlyCamera = 0.8
        propOnlyIso = 1.0

        #list all datasets
        dataset_info = np.genfromtxt(descriptor_file, skip_header=1, dtype=None)
        num_files = dataset_info.shape[0]
        print('Datasets:')
        for j in range(num_files):
            name = str(dataset_info[j][0].decode('ascii'))
            min_iso = float(dataset_info[j][1])
            max_iso = float(dataset_info[j][2])
            print(name,"  iso=[%f,%f]"%(min_iso, max_iso))

        expected_filesize = SIZE * SIZE * 10 * T * num_files * numImages * 4
        print("Shape: B=%d, T=%d, C=%d, H=%d, W=%d"%(num_files*numImages, T, 10, SIZE, SIZE))
        print("Expected filesize:", humanbytes(expected_filesize))

        sample_index = 0
        sample_index2 = 0
        for j in range(num_files):
            name = str(dataset_info[j][0].decode('ascii'))

            inputFile = os.path.abspath(os.path.join(os.path.dirname(descriptor_file), name))
            print("Process", inputFile)
            if not generator.loadVolume(inputFile):
                print("Unable to load volume")
                continue
            if inputMipmapLevel>0:
                print("Create mipmap level")
                torch.ops.renderer.create_mipmap_level(inputMipmapLevel, inputMipmapFilter)
                
            #load transfer functions
            transfer_functions = glob.glob(tfDirectory + "/" + name[name.rfind('/') + 1:name.rfind('.')] + "/*.tf")
        
            if len(transfer_functions) == 0:
                print("There is no TF created for this volume.")
                exit(1)

            pg = ProgressBar(numImages, 'Render', length=50)
            i = 0
            numAttempts = 0
            while i < numImages:
                pg.print_progress_bar(i)
                numAttempts += 1
                if numAttempts > 5 * numImages:
                    print("Failed to sample enough images for the current volume, running out of attempts")
                    if save_config_file is not None:
                        for i in range(numImages - i):
                            config_file.write(
                                "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\n"%(
                                    False, 0, 0, 0,
                                    0,0,0, 0,0,0, 0,0,0, 0,0,0,
                                    0,0, 0,0,0, 0,0, 0
                                    ))
                    break;

                if restore_config_file is None:
                    originStart = GroundTruthDatasetGenerator.randomPointOnSphere() * \
                        GroundTruthDatasetGenerator.randomFloat(camera_distance[0], camera_distance[1])
                    originStart = origin_mapping(originStart)
                    lookAtStart = GroundTruthDatasetGenerator.randomPointOnSphere() * 0.1
                    lookAtStart = lookAt_mapping(lookAtStart)
                    while True:
                        originEnd = GroundTruthDatasetGenerator.randomPointOnSphere() * \
                            GroundTruthDatasetGenerator.randomFloat(camera_distance[0], camera_distance[1])
                        originEnd = origin_mapping(originEnd)
                        if np.linalg.norm(originEnd - originStart) < maxDist:
                            break
                    lookAtEnd = GroundTruthDatasetGenerator.randomPointOnSphere() * 0.1
                    lookAtEnd = lookAt_mapping(lookAtEnd)
                    up = GroundTruthDatasetGenerator.randomPointOnSphere()
                    up = up_mapping(up)

                    settings[GroundTruthDatasetGenerator.CAM_UP] = list(up)
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_START] = list(originStart)
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_END] = list(originEnd)
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_START] = list(lookAtStart)
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_END] = list(lookAtEnd)

                    chosen_tf_idx = np.random.randint(len(transfer_functions))
                    tf_json = load_tf_v2(transfer_functions[chosen_tf_idx])
                    settings[GroundTruthDatasetGenerator.DENSITY_AXIS_OPACITY] = tf_json["densityAxisOpacity"]
                    settings[GroundTruthDatasetGenerator.OPACITY_AXIS] = tf_json["opacityAxis"]
                    settings[GroundTruthDatasetGenerator.DENSITY_AXIS_COLOR] = tf_json["densityAxisColor"]
                    settings[GroundTruthDatasetGenerator.COLOR_AXIS] = tf_json["colorAxis"]
                    settings[GroundTruthDatasetGenerator.MIN_DENSITY] = tf_json["minDensity"]
                    settings[GroundTruthDatasetGenerator.MAX_DENSITY] = tf_json["maxDensity"]
                    
                    opacity = opacity_scaling[0] + \
                        np.random.rand()*(opacity_scaling[1]-opacity_scaling[0])
                    settings[GroundTruthDatasetGenerator.OPACITY_SCALING] = opacity

                    useShading = np.random.randint(2)
                    specularExponent = int(2**np.random.randint(2,5))
                    lightDirection = np.array([
                        np.random.rand()*2-1,
                        np.random.rand()*2-1,
                        1])
                    lightDirection = list(lightDirection / np.linalg.norm(lightDirection))
                    settings[GroundTruthDatasetGenerator.DVR_USE_SHADING] = useShading
                    settings[GroundTruthDatasetGenerator.SPECULAR_EXPONENT] = specularExponent
                    settings[GroundTruthDatasetGenerator.LIGHT_DIRECTION] = lightDirection
                    settings[GroundTruthDatasetGenerator.VALUE_SCALING] = None

                    output = generator.render(settings)
                    if output is not None:
                        dset.resize(sample_index+1, axis=0)
                        dset[sample_index,...] = output
                        sample_index += 1
                        i += 1

                        if save_config_file is not None:
                            valueScaling = settings[GroundTruthDatasetGenerator.VALUE_SCALING]
                            config_file.write(
                                "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\n"%(
                                    True, up[0], up[1], up[2],
                                    originStart[0], originStart[1], originStart[2],
                                    originEnd[0], originEnd[1], originEnd[2],
                                    lookAtStart[0], lookAtStart[1], lookAtStart[2],
                                    lookAtEnd[0], lookAtEnd[1], lookAtEnd[2],
                                    useShading, specularExponent,
                                    lightDirection[0], lightDirection[1], lightDirection[2],
                                    chosen_tf_idx, opacity, valueScaling
                                    ))

                else:
                    # use configuration from the settings file
                    cfg = config[sample_index2,:]
                    valid = cfg[0]; cfg = cfg[1:]
                    if not valid:
                        sample_index2 += 1
                        i += 1
                        continue
                    settings[GroundTruthDatasetGenerator.CAM_UP] = [cfg[0], cfg[1], cfg[2]]
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_START] = [cfg[3], cfg[4], cfg[5]]
                    settings[GroundTruthDatasetGenerator.CAM_ORIGIN_END] = [cfg[6], cfg[7], cfg[8]]
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_START] = [cfg[9], cfg[10], cfg[11]]
                    settings[GroundTruthDatasetGenerator.CAM_LOOKAT_END] = [cfg[12], cfg[13], cfg[14]]
                    settings[GroundTruthDatasetGenerator.DVR_USE_SHADING] = int(cfg[15])
                    settings[GroundTruthDatasetGenerator.SPECULAR_EXPONENT] = int(cfg[16])
                    settings[GroundTruthDatasetGenerator.LIGHT_DIRECTION] = [cfg[17], cfg[18], cfg[19]]
                    settings[GroundTruthDatasetGenerator.OPACITY_SCALING] = int(cfg[21])
                    settings[GroundTruthDatasetGenerator.VALUE_SCALING] = int(cfg[22])

                    chosen_tf_idx = int(cfg[20])
                    tf_json = load_tf_v2(transfer_functions[chosen_tf_idx])
                    settings[GroundTruthDatasetGenerator.DENSITY_AXIS_OPACITY] = tf_json["densityAxisOpacity"]
                    settings[GroundTruthDatasetGenerator.OPACITY_AXIS] = tf_json["opacityAxis"]
                    settings[GroundTruthDatasetGenerator.DENSITY_AXIS_COLOR] = tf_json["densityAxisColor"]
                    settings[GroundTruthDatasetGenerator.COLOR_AXIS] = tf_json["colorAxis"]
                    settings[GroundTruthDatasetGenerator.MIN_DENSITY] = tf_json["minDensity"]
                    settings[GroundTruthDatasetGenerator.MAX_DENSITY] = tf_json["maxDensity"]

                    output = generator.render(settings, ignoreMinSamples=True)
                    assert output is not None
                    dset.resize(sample_index+1, axis=0)
                    dset[sample_index,...] = output
                    sample_index += 1
                    sample_index2 += 1
                    i += 1

            pg.print_progress_bar(numImages)

def generateGroundTruthIso():
    #TODO: interpolate flow during generation
    gen = GroundTruthDatasetGenerator()
    gen.loadLibrary("./Renderer.dll")

    plotting = False
    if plotting:
        plt.ion()

    aoSamples = 256
    inputMipmapLevel = 0
    inputMipmapFilter = "average"
    save_config_file = None
    restore_config_file = None
    downsampling_factor = 1

    # EJECTA
    if 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        numImages = 20
        seed = 1
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-noAO.hdf5'
        save_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-screen4x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        downsampling_factor = 4
        numImages = 20
        seed = 1
        aoSamples = 0
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-screen8x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        downsampling_factor = 8
        numImages = 20
        seed = 1
        aoSamples = 0
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-obj2x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        inputMipmapLevel = 1
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-obj4x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        inputMipmapLevel = 3
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-obj8x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        inputMipmapLevel = 7
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-test.hdf5'
        save_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-test-cfg.txt'
        numImages = 20
        seed = 2
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-test-screen4x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-ejecta-v2-test-cfg.txt'
        numImages = 20
        seed = 2
        aoSamples = 0
        downsampling_factor = 4
    # RM
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseRM.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-rm.hdf5'
        save_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-rm-cfg.txt'
        numImages = 20
        seed = 1
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseRM.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-rm-screen4x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-rm-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 4
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseRM.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-rm-screen8x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-rm-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 8
    # HUMAN
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseHuman.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-human-v1.hdf5'
        numImages = 20
        seed = 1
    # THORAX
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseThorax.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-thorax.hdf5'
        save_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-thorax-cfg.txt'
        numImages = 20
        seed = 1
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseThorax.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-thorax-screen4x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-thorax-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 4
    elif 1:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseThorax.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-thorax-screen8x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-thorax-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 8
    # HEAD
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseHead.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-head.hdf5'
        save_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-head-cfg.txt'
        numImages = 20
        seed = 1
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseHead.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-head-screen4x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-head-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 4
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseHead.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-head-screen8x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-head-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 8
    # CUBE
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsCube2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-cube2.hdf5'
        save_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-cube2-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
    elif 0:
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsCube2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-rendering-cube2-screen8x.hdf5'
        restore_config_file = 'D:/VolumeSuperResolution-InputData/gt-rendering-cube2-cfg.txt'
        numImages = 20
        seed = 1
        aoSamples = 0
        downsampling_factor = 8

    np.random.seed(seed)
    generateRandomSamplesIso(
        gen, descriptorFile, 
        outputFile, numImages, 
        aoSamples, inputMipmapLevel, inputMipmapFilter,
        save_config_file, restore_config_file,
        downsampling_factor)

def generateGroundTruthDvr():
    #TODO: interpolate flow during generation
    gen = GroundTruthDatasetGenerator()
    gen.loadLibrary("./Renderer.dll")

    plotting = False
    if plotting:
        plt.ion()

    inputMipmapLevel = 0
    inputMipmapFilter = "average"
    save_config_file = None
    restore_config_file = None
    downsampling_factor = 1
    step_size = 0.1
    opacity_scaling = [20, 100]
    resolution = 512

    origin_mapping = lambda x : x
    lookAt_mapping = lambda x : x
    up_mapping = lambda x : x

    Cfg = namedtuple("Cfg", 
                     "descriptor, output, cfgfile, save, numImages, seed, downsampling, mipmap, stepsize, dsetName")

    
    if 0: # EJECTA
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense3.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6%s.hdf5'
        config_file = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6-cfg.txt'
        tfDirectory = "../../isosurface-super-resolution-data/volumes/tfs-filtered-user/"
        numImages = 20
        seed = 1
        opacity_scaling = [150, 250]
        cfgs = [
            Cfg(descriptorFile, outputFile%"", config_file, True, numImages, seed, 1, 0, step_size, 'gt'), # base
            Cfg(descriptorFile, outputFile%"-screen4x", config_file, False, numImages, seed, 4, 0, step_size, 'gt'), # screen
            Cfg(descriptorFile, outputFile%"-screen8x", config_file, False, numImages, seed, 8, 0, step_size, 'gt'),
            Cfg(descriptorFile, outputFile%"-obj2x", config_file, False, numImages, seed, 1, 1, step_size, 'gt'), # obj
            Cfg(descriptorFile, outputFile%"-obj4x", config_file, False, numImages, seed, 1, 3, step_size, 'gt'),
            Cfg(descriptorFile, outputFile%"-step10", config_file, False, numImages, seed, 1, 0, 1.0, 'gt'), # step
            Cfg(descriptorFile, outputFile%"-screen4x-step10", config_file, False, numImages, seed, 4, 0, 1.0, 'gt'),
            ]
    elif 0: # EJECTA 2
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta4%s.hdf5'
        config_file = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta4-cfg.txt'
        tfDirectory = "../../isosurface-super-resolution-data/volumes/tfs-filtered/"
        numImages = 10
        seed = 1
        resolution = 1024
        cfgs = [
            Cfg(descriptorFile, outputFile%"", config_file, True, numImages, seed, 1, 0, step_size, 'gt'), # base
            #Cfg(descriptorFile, outputFile%"-screen4x", config_file, False, numImages, seed, 4, 0, step_size, 'gt'), # screen
            #Cfg(descriptorFile, outputFile%"-screen8x", config_file, False, numImages, seed, 8, 0, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-obj2x", config_file, False, numImages, seed, 1, 1, step_size, 'gt'), # obj
            #Cfg(descriptorFile, outputFile%"-obj4x", config_file, False, numImages, seed, 1, 3, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-step10", config_file, False, numImages, seed, 1, 0, 1.0, 'gt'), # step
            #Cfg(descriptorFile, outputFile%"-screen4x-step10", config_file, False, numImages, seed, 4, 0, 1.0, 'gt'),
            ]
    elif 0: # Ejecta Adaptive Stepsize
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta5%s.hdf5'
        config_file = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta5-cfg.txt'
        tfDirectory = "../../isosurface-super-resolution-data/volumes/tfs-filtered/"
        numImages = 10
        seed = 1
        resolution = 512
        step_sizes = range(-5, +3)
        #cfgs = [Cfg(descriptorFile, outputFile%"-steps", config_file, False, 
        #            numImages, seed, 1, 0, 2**step_size, 'step%s'%step_size)
        #        for step_size in step_sizes]
        cfgs = [
            #Cfg(descriptorFile, outputFile%"-input1x", config_file, False, numImages, seed, 1, 0, 1.0, 'gt'),
            #Cfg(descriptorFile, outputFile%"-input2x", config_file, False, numImages, seed, 2, 0, 1.0, 'gt')
            Cfg(descriptorFile, outputFile%"-gt", config_file, False, numImages, seed, 1, 0, 0.1, 'gt')
            ]
    elif 0: # EJECTA Test
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDense3.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6-test%s.hdf5'
        config_file = 'D:/VolumeSuperResolution-InputData/gt-dvr-ejecta6-test-cfg.txt'
        tfDirectory = "../../isosurface-super-resolution-data/volumes/tfs-filtered-user/"
        numImages = 10
        seed = 2
        opacity_scaling = [150, 250]
        cfgs = [
            Cfg(descriptorFile, outputFile%"", config_file, True, numImages, seed, 1, 0, step_size, 'gt'), # base
            #Cfg(descriptorFile, outputFile%"-screen4x", config_file, False, numImages, seed, 4, 0, step_size, 'gt'), # screen
            Cfg(descriptorFile, outputFile%"-screen8x", config_file, False, numImages, seed, 8, 0, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-obj2x", config_file, False, numImages, seed, 1, 1, step_size, 'gt'), # obj
            #Cfg(descriptorFile, outputFile%"-obj4x", config_file, False, numImages, seed, 1, 3, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-step10", config_file, False, numImages, seed, 1, 0, 1.0, 'gt'), # step
            #Cfg(descriptorFile, outputFile%"-screen4x-step10", config_file, False, numImages, seed, 4, 0, 1.0, 'gt'),
            ]
    elif 0: # RM Test
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseRM2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-dvr-rm2-test%s.hdf5'
        config_file = 'D:/VolumeSuperResolution-InputData/gt-dvr-rm2-test-cfg.txt'
        tfDirectory = "../../isosurface-super-resolution-data/volumes/tfs-filtered-user/"
        #descriptorFile = 'D:/Promotion/isosurface-data/inputsDenseRM2.dat'
        #outputFile = 'D:/Promotion/isosurface-data/inputs/gt-dvr-rm1-test%s.hdf5'
        #config_file = 'D:/Promotion/isosurface-data/inputs/gt-dvr-rm1-test-cfg.txt'
        #tfDirectory = "D:/Promotion/isosurface-data/tfs-filtered-user/"
        numImages = 10
        seed = 3
        resolution = 1024
        opacity_scaling = [150, 300]
        origin_mapping = lambda x : np.array([0.7*x[0], 0.7*x[1], -abs(0.7*x[2])-0.1])
        lookAt_mapping = lambda x : np.array([x[0], x[1], x[2]-0.1])
        def up_mapping_fun(x):
            x0 = 0.5*x[0]
            x1 = 0.5*x[1]
            x2 = 0.5*abs(x[2])+1
            f = 1/np.sqrt(x0*x0+x1*x1+x2*x2)
            return np.array([x0*f, x1*f, x2*f])
        up_mapping = up_mapping_fun
        cfgs = [
            Cfg(descriptorFile, outputFile%"", config_file, True, numImages, seed, 1, 0, step_size, 'gt'), # base
            #Cfg(descriptorFile, outputFile%"-screen4x", config_file, False, numImages, seed, 4, 0, step_size, 'gt'), # screen
            Cfg(descriptorFile, outputFile%"-screen8x", config_file, False, numImages, seed, 8, 0, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-obj2x", config_file, False, numImages, seed, 1, 1, step_size, 'gt'), # obj
            #Cfg(descriptorFile, outputFile%"-obj4x", config_file, False, numImages, seed, 1, 3, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-step10", config_file, False, numImages, seed, 1, 0, 1.0, 'gt'), # step
            #Cfg(descriptorFile, outputFile%"-screen4x-step10", config_file, False, numImages, seed, 4, 0, 1.0, 'gt'),
            ]
    elif 1: # Thorax Test
        descriptorFile = '../../isosurface-super-resolution-data/volumes/inputsDenseThorax2.dat'
        outputFile = 'D:/VolumeSuperResolution-InputData/gt-dvr-thorax2-test%s.hdf5'
        config_file = 'D:/VolumeSuperResolution-InputData/gt-dvr-thorax2-test-cfg.txt'
        tfDirectory = "../../isosurface-super-resolution-data/volumes/tfs-filtered-user/"
        #descriptorFile = 'D:/Promotion/isosurface-data/inputsDenseThorax2.dat'
        #outputFile = 'D:/Promotion/isosurface-data/inputs/gt-dvr-thorax1-test%s.hdf5'
        #config_file = 'D:/Promotion/isosurface-data/inputs/gt-dvr-thorax1-test-cfg.txt'
        #tfDirectory = "D:/Promotion/isosurface-data/tfs-filtered-user/"
        numImages = 10
        seed = 2
        resolution = 1024
        opacity_scaling = [150, 250]
        origin_mapping = lambda x : np.array([1.5*x[0], 1.5*x[1], -abs(0.5*x[2])])
        lookAt_mapping = lambda x : x
        def up_mapping_fun(x):
            x0 = 0.1*x[0]
            x1 = 0.1*x[1]
            x2 = 0.1*abs(x[2])+1
            f = 1/np.sqrt(x0*x0+x1*x1+x2*x2)
            return np.array([x0*f, x1*f, x2*f])
        cfgs = [
            Cfg(descriptorFile, outputFile%"", config_file, True, numImages, seed, 1, 0, step_size, 'gt'), # base
            #Cfg(descriptorFile, outputFile%"-screen4x", config_file, False, numImages, seed, 4, 0, step_size, 'gt'), # screen
            Cfg(descriptorFile, outputFile%"-screen8x", config_file, False, numImages, seed, 8, 0, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-obj2x", config_file, False, numImages, seed, 1, 1, step_size, 'gt'), # obj
            #Cfg(descriptorFile, outputFile%"-obj4x", config_file, False, numImages, seed, 1, 3, step_size, 'gt'),
            #Cfg(descriptorFile, outputFile%"-step10", config_file, False, numImages, seed, 1, 0, 1.0, 'gt'), # step
            #Cfg(descriptorFile, outputFile%"-screen4x-step10", config_file, False, numImages, seed, 4, 0, 1.0, 'gt'),
            ]

    for cfg in cfgs:
        print("\n=== Process", cfg, "===\n")

        np.random.seed(cfg.seed)
        generateRandomSamplesDVR(
            gen, cfg.descriptor, tfDirectory,
            cfg.output, cfg.numImages, 
            cfg.mipmap, inputMipmapFilter,
            cfg.cfgfile if cfg.save==True else None, cfg.cfgfile if cfg.save==False else None,
            cfg.downsampling, cfg.stepsize, opacity_scaling, resolution=resolution, dset_name=cfg.dsetName,
            origin_mapping = origin_mapping, lookAt_mapping = lookAt_mapping, up_mapping=up_mapping)

if __name__ == "__main__":
    #generateGroundTruthIso()
    generateGroundTruthDvr()
