import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tkinter import filedialog, Tk
import json
import os.path
import traceback
import numpy as np

import models
import importance

class EnhanceNetWrapper(nn.Module):
    def __init__(self, orig):
        super().__init__();
        self._orig = orig
    def forward(self, input, mask=None):
        return self._orig(input)[0]
class NormalizedImportanceNetwork(torch.nn.Module):
    def __init__(self, net, norm, denorm):
        super().__init__()
        self._net = net
        self._norm = norm
        self._denorm = denorm
    def forward(self, x):
        x = self._norm(x)
        y = self._net(x)
        return y
class NormalizedReconstructionNetwork(torch.nn.Module):
    def __init__(self, net, norm, denorm):
        super().__init__()
        self._net = net
        self._norm = norm
        self._denorm = denorm
    def forward(self, x, mask):
        x = self._norm(x)
        y = self._net(x, mask)
        y = self._denorm(y)
        return y

class BaselineReconModel(nn.Module):
    def __init__(self, selectedChannelsOut):
        super().__init__();
        self._selectedChannelsOut = selectedChannelsOut
    def forward(self, input):
        return input[:, self._selectedChannelsOut, :, :]

def convert_unshaded(checkpoint, parameters, output_filename):
    print("Convert dense unshaded super-resolution network")

    if 'upscaleFactor' in parameters:
        upscale_factor = parameters['upscaleFactor']
    else:
        upscale_factor = parameters.get('upscale_factor', 4)
    initial_image = parameters.get('initialImage', 'zero')
    print("upscale factor:", upscale_factor)
    print("initial image:", initial_image)

    model = checkpoint['model']
    device = torch.device('cuda')
    model.to(device)
    model.train(False)
    print("Model:")
    print(model)

    #find first module
    first_module = model
    while True:
        it = first_module.children()
        try:
            o = next(it)
        except StopIteration:
            break
        first_module = o
    input_channels = first_module.in_channels
    print("number of input channels:", input_channels)
    print("expected 5 + 6 * upscale_factor^2: ", 5+6*(upscale_factor**2))

    print("Convert to script")
    try:
        #scripted_module = torch.jit.script(model)
        input = torch.rand(1, input_channels, 128, 128, dtype=torch.float32, device=device)
        scripted_module = torch.jit.trace(model, input)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return

    settings = {
        'upscale_factor' : upscale_factor,
        'initial_image' : initial_image,
        'input_channels' : input_channels
        }
    settings_json = json.dumps(settings)

    print("Save to", output_filename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(scripted_module, output_filename, _extra_files=extra_files)

def convert_importance(checkpoint, parameters, output_filename):
    print("Convert importance sampling network")

    networkUpscale = parameters['networkUpscale']
    postUpscale = parameters['postUpscale']
    upscale_factor = networkUpscale * postUpscale
    initial_image = parameters['initialImage']
    disableTemporal = parameters['disableTemporal']
    outputLayer = parameters['outputLayer']
    print("upscale factor:", upscale_factor)
    print("initial image:", initial_image)

    model = checkpoint['model']
    device = torch.device('cuda')
    model.to(device)
    model.train(False)
    print("Model:")
    print(model)

    #find first module
    first_module = model
    while True:
        it = first_module.children()
        try:
            o = next(it)
        except StopIteration:
            break
        first_module = o
    input_channels = first_module.in_channels
    print("number of input channels:", input_channels)
    if input_channels == 5:
        hasPreviousInput = False
    elif input_channels == 5 + upscaleFactor**2:
        hasPreviousInput = True
    else:
        print("illegal input channel count:", input_channels)
    print("has previous input:", hasPreviousInput)

    print("Convert to script")
    try:
        #scripted_module = torch.jit.script(model)
        input = torch.rand(1, input_channels, 128, 128, dtype=torch.float32, device=device)
        scripted_module = torch.jit.trace(model, input)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return

    settings = {
        'networkUpscale' : networkUpscale,
        'postUpscale' : postUpscale,
        'upscale_factor' : upscale_factor,
        'input_channels' : input_channels,
        'initial_image' : initial_image,
        'disableTemporal' : disableTemporal,
        'outputLayer' : outputLayer
        }
    settings_json = json.dumps(settings)

    print("Save to", output_filename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(scripted_module, output_filename, _extra_files=extra_files)

def convert_sparse(checkpoint, parameters, output_filename):
    print("Convert sparse unshaded super-resolution network")
    import models

    # create model
    externalFlow = parameters.get('externalFlow', False)
    if externalFlow:
        input_channels = 7
        output_channels = 6
    else:
        input_channels = 9
        output_channels = 8
    input_channels_with_previous = input_channels + output_channels
    model = models.UNet(input_channels_with_previous, output_channels,
                        parameters['depth'], parameters['filters'], parameters['padding'],
                        parameters['batchNorm'], parameters['residual'],
                        parameters['hardInput'], parameters['upMode'], True)

    # restore weights
    model.load_state_dict(checkpoint['model_params'], True)
    device = torch.device('cuda')
    model.to(device)
    model.train(False)
    print("Model:")
    print(model)

    print("Convert to script")
    try:
        def genInput(width, height):
            input = torch.rand(1, input_channels_with_previous, height, width, dtype=torch.float32, device=device)
            mask = (torch.rand(1, 1, height, width, dtype=torch.float32, device=device) > 0.5).to(torch.float32)
            return (input, mask)
        inputs = [
            genInput(128, 128),
            genInput(262, 913)]#,
            #genInput(841, 498),
            #genInput(713, 582)]
        print("Dry run:")
        for input in inputs:
            print("====== Check input of size", input[0].shape, "======")
            run1 = model(*input)
            assert input[0].shape[-2:] == run1[0].shape[-2:], "shapes don't match"
        print("Trace run:")
        scripted_module = torch.jit.trace(model, inputs[0])#, check_inputs=inputs)
        #scripted_module = torch.jit.script(model)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return

    settings = parameters
    settings_json = json.dumps(settings)

    print("Save to", output_filename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(scripted_module, output_filename, _extra_files=extra_files)

def convert_adaptive(checkpoint, parameters, output_filename):
    """
    Convert both importance + reconstruction network
    """

    # channels
    mode = parameters['mode']
    inputChannels = []
    if mode == 'iso':
        inputChannelsStr = (parameters.get('inputChannels', None) or 'normal,depth').split(',') + ["mask"]
        inputChannels += [('mask', 0)]
        if 'normal' in inputChannelsStr:
            inputChannels += [('normalX', 1), ('normalY', 2), ('normalZ', 3)]
        if 'depth' in inputChannelsStr:
            inputChannels += [('depth', 4)]
        channels_low = len(inputChannels)
        channels_sampling = channels_low + 2
        channels_out = 6 # mask, normal x, normal y, normal z, depth, ao
    elif mode == 'dvr':
        inputChannelsStr = (parameters.get('inputChannels', None) or 'color').split(',')
        channels_low = 0
        if 'color' in inputChannelsStr:
            inputChannels += [('red', 0), ('green', 1), ('blue', 2), ('alpha', 3)]
        if 'normal' in inputChannelsStr:
            inputChannels += [('normalX', 4), ('normalY', 5), ('normalZ', 6)]
        if 'depth' in inputChannelsStr:
            inputChannels += [('depth', 7)]
        channels_low = len(inputChannels)
        channels_sampling = channels_low + 1 # + samples
        channels_out = 4 # rgba
    else:
        raise ValueError("unknown mode "+mode)
    print("Input channels:", inputChannels)
    channels_low_with_previous = channels_low + parameters['importanceNetUpscale']**2
    channels_sampling_with_previous = channels_sampling + channels_out
    device = torch.device('cuda')

    # convert importance model to script
    print("Convert importance sampling network to script")
    importanceModel = checkpoint['importanceModel']
    importanceRequiresPrevious = parameters["type"] == "adaptive2"
    if importanceRequiresPrevious:
        importanceInputSize = (1, channels_low_with_previous, 256, 256)
    else:
        importanceInputSize = (1, channels_low, 256, 256)

    # HACK, in early trainings, ._residual_net did not properly derive from Network!
    # we have to create it anew
    if hasattr(importanceModel, '_residual_net'):
        importanceModel._residual_net = importance.GradientImportanceMap(
            importanceModel._upsampleFactor, *importanceModel._residual_net._channels)

    importanceModel.to(device)
    importanceModel.train(False)
    try:
        input = torch.rand(*importanceInputSize, dtype=torch.float32, device=device)
        importanceScriptedModel = torch.jit.trace(importanceModel, input)
        #importanceScriptedModel = torch.jit.script(importanceModel)
        # test other input
        input_size2 = (importanceInputSize[0], importanceInputSize[1], 218, 170)
        input = torch.rand(*input_size2, dtype=torch.float32, device=device)
        output = importanceScriptedModel(input)
        print("Importance output shape:", output.shape)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return
    importanceSettings = {
        'mode' : mode,
        'networkUpscale' : parameters['importanceNetUpscale'],
        'postUpscale' : parameters['importancePostUpscale'],
        'upscale_factor' : parameters['importanceNetUpscale'] * parameters['importancePostUpscale'],
        'input_channel_count' : channels_low,
        'input_channels' : inputChannels,
        'initial_image' : 'zero',
        'disableTemporal' : parameters.get("importanceDisableTemporal", True),
        'requiresPrevious': importanceRequiresPrevious,
        'outputLayer' : parameters['importanceOutputLayer']
        }
    settings_json = json.dumps(importanceSettings)
    importanceOutputFilename = output_filename[:-3] + "_importance.pt"
    print("Save to", importanceOutputFilename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(importanceScriptedModel, importanceOutputFilename, _extra_files=extra_files)

    # convert reconstruction model to script
    print("Convert reconstruction network to script")
    reconstructionModel = checkpoint['reconstructionModel']
    reconstructionInputSize = (1, channels_sampling_with_previous, 256, 256)
    reconstructionModel.to(device)
    reconstructionModel.train(False)
    try:
        input = torch.rand(*reconstructionInputSize, dtype=torch.float32, device=device)
        inputMask = torch.rand((1, 1, 256, 256), dtype=torch.float32, device=device)
        reconstructionScriptedModel = torch.jit.trace(reconstructionModel, (input, inputMask))
        #reconstructionScriptedModel = torch.jit.script(reconstructionModel)
        # test other input
        input_size2 = (reconstructionInputSize[0], reconstructionInputSize[1], 218, 170)
        input = torch.rand(*input_size2, dtype=torch.float32, device=device)
        inputMask = torch.rand((1, 1, 218, 170), dtype=torch.float32, device=device)
        _ = reconstructionScriptedModel(input, inputMask)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return
    reconstructionSettings = {
        'mode' : mode,
        'interpolateInput' : parameters.get("reconInterpolateInput", False),
        'residual' : parameters.get("reconResidual", False),
        'hardInput' : parameters['reconHardInput'],
        'externalFlow' : True,
        'architecture' : parameters['reconModel'],
        'depth' : parameters['reconLayers'],
        'filters' : parameters['reconFilters'],
        'padding' : 'zero',
        'expectMask' : True,
        'input_channel_count' : channels_sampling,
        'input_channels' : inputChannels,
        }
    settings_json = json.dumps(reconstructionSettings)
    reconstructionOutputFilename = output_filename[:-3] + "_recon.pt"
    print("Save to", reconstructionOutputFilename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(reconstructionScriptedModel, reconstructionOutputFilename, _extra_files=extra_files)

def convert_dense_dvr(checkpoint, parameters, output_filename):
    print("Convert dense dvr super-resolution network")

    if 'upscaleFactor' in parameters:
        upscale_factor = parameters['upscaleFactor']
    else:
        upscale_factor = parameters.get('upscale_factor', 4)
    initial_image = parameters.get('initialImage', 'zero')
    print("upscale factor:", upscale_factor)
    print("initial image:", initial_image)

    model = checkpoint['model']
    device = torch.device('cuda')
    model.to(device)
    model.train(False)
    print("Model:")
    print(model)

    #input channels
    inputChannelsString = parameters.get("inputChannels", "color")
    selectedChannelsList = inputChannelsString.split(',')
    selectedChannels = []
    receives_normal = False
    receives_depth = False
    if "color" in selectedChannelsList:
        selectedChannels += [('red', 0), ('green', 1), ('blue', 2), ('alpha', 3)]
    if "normal" in selectedChannelsList:
        selectedChannels += [('normalX', 4), ('normalY', 5), ('normalZ', 6)]
        receives_normal = True
    if "depth" in selectedChannelsList:
        selectedChannels += [('depth', 7)]
        receives_depth = True
    input_channels_len = len(selectedChannels)

    #find first module
    first_module = model
    while True:
        it = first_module.children()
        try:
            o = next(it)
        except StopIteration:
            break
        first_module = o
    input_channels = first_module.in_channels
    print("number of input channels:", input_channels)
    print("expected", input_channels_len, "+ 4 * upscale_factor^2: ", input_channels_len+4*(upscale_factor**2))

    print("Convert to script")
    try:
        #scripted_module = torch.jit.script(model)
        input = torch.rand(1, input_channels, 128, 128, dtype=torch.float32, device=device)
        scripted_module = torch.jit.trace(model, input)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return

    settings = {
        'upscale_factor' : upscale_factor,
        'initial_image' : initial_image,
        'input_channels' : input_channels,
        'input_channels_list' : selectedChannelsList,
        'receives_normal': receives_normal,
        'receives_depth': receives_normal
        }
    settings_json = json.dumps(settings)

    print("Save to", output_filename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(scripted_module, output_filename, _extra_files=extra_files)

def convert_stepsize(checkpoint, parameters, output_filename):
    """
    Convert both importance + reconstruction network
    """

    # channels
    mode = parameters['mode']
    inputChannels = []
    if mode == 'iso':
        inputChannelsStr = (parameters.get('inputChannels', None) or 'normal,depth').split(',') + ["mask"]
        inputChannels += [('mask', 0)]
        if 'normal' in inputChannelsStr:
            inputChannels += [('normalX', 1), ('normalY', 2), ('normalZ', 3)]
        if 'depth' in inputChannelsStr:
            inputChannels += [('depth', 4)]
        channels_low = len(inputChannels)
        channels_sampling = channels_low
        channels_out = 6 # mask, normal x, normal y, normal z, depth, ao
    elif mode == 'dvr':
        inputChannelsStr = (parameters.get('inputChannels', None) or 'color').split(',')
        channels_low = 0
        if 'color' in inputChannelsStr:
            inputChannels += [('red', 0), ('green', 1), ('blue', 2), ('alpha', 3)]
        if 'normal' in inputChannelsStr:
            inputChannels += [('normalX', 4), ('normalY', 5), ('normalZ', 6)]
        if 'depth' in inputChannelsStr:
            inputChannels += [('depth', 7)]
        channels_low = len(inputChannels)
        channels_sampling = channels_low
        channels_out = 4 # rgba
    else:
        raise ValueError("unknown mode "+mode)
    print("Input channels:", inputChannels)
    channels_low_with_previous = channels_low + parameters['importanceUpscale']**2
    channels_sampling_with_previous = channels_sampling + channels_out
    device = torch.device('cuda')

    # convert importance model to script
    print("Convert stepsize network to script")
    importanceModel = checkpoint['importanceModel']
    importanceRequiresPrevious = True
    if importanceRequiresPrevious:
        importanceInputSize = (1, channels_low_with_previous, 256, 256)
    else:
        importanceInputSize = (1, channels_low, 256, 256)

    importanceModel.to(device)
    importanceModel.train(False)
    try:
        input = torch.rand(*importanceInputSize, dtype=torch.float32, device=device)
        importanceScriptedModel = torch.jit.trace(importanceModel, input)
        #importanceScriptedModel = torch.jit.script(importanceModel)
        # test other input
        input_size2 = (importanceInputSize[0], importanceInputSize[1], 218, 170)
        input = torch.rand(*input_size2, dtype=torch.float32, device=device)
        output = importanceScriptedModel(input)
        print("Importance output shape:", output.shape)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return
    importanceSettings = {
        'mode' : mode,
        'networkUpscale' : parameters['importanceUpscale'],
        'postUpscale' : 1,
        'input_channel_count' : channels_low,
        'input_channels' : inputChannels,
        'initial_image' : 'zero',
        'disableTemporal' : parameters["importanceDisableTemporal"],
        'requiresPrevious': importanceRequiresPrevious,
        'outputLayer' : parameters['importanceOutputLayer']
        }
    settings_json = json.dumps(importanceSettings)
    importanceOutputFilename = output_filename[:-3] + "_importance.pt"
    print("Save to", importanceOutputFilename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(importanceScriptedModel, importanceOutputFilename, _extra_files=extra_files)

    # convert reconstruction model to script
    print("Convert reconstruction network to script")
    reconstructionModel = checkpoint['reconstructionModel']
    reconstructionInputSize = (1, channels_sampling_with_previous, 256, 256)
    reconstructionModel.to(device)
    reconstructionModel.train(False)
    try:
        input = torch.rand(*reconstructionInputSize, dtype=torch.float32, device=device)
        reconstructionScriptedModel = torch.jit.trace(reconstructionModel, input)
        #reconstructionScriptedModel = torch.jit.script(reconstructionModel)
        # test other input
        input_size2 = (reconstructionInputSize[0], reconstructionInputSize[1], 218, 170)
        input = torch.rand(*input_size2, dtype=torch.float32, device=device)
        _ = reconstructionScriptedModel(input)
    except Exception as ex:
        print("Unable to convert:")
        print(traceback.format_exc())
        return
    reconstructionSettings = {
        'mode' : mode,
        'residual' : parameters["reconResidual"],
        'architecture' : parameters['reconModel'],
        'depth' : parameters['reconLayers'],
        'filters' : parameters['reconFilters'],
        'padding' : 'zero',
        'input_channel_count' : channels_sampling,
        'input_channels' : inputChannels,
        }
    settings_json = json.dumps(reconstructionSettings)
    reconstructionOutputFilename = output_filename[:-3] + "_recon.pt"
    print("Save to", reconstructionOutputFilename)
    extra_files = torch._C.ExtraFilesMap()
    extra_files['settings.json'] = settings_json
    print(extra_files)
    torch.jit.save(reconstructionScriptedModel, reconstructionOutputFilename, _extra_files=extra_files)

def convert(input_filename, output_filename):
    print("Load", input_filename)
    print("And save to", output_filename)

    checkpoint = torch.load(input_filename)
    
    parameters = checkpoint.get('parameters', dict())
    print("Parameters:")
    for key,value in parameters.items():
        print("  %s=%s"%(key, value))
    if not isinstance(parameters, dict):
        parameters = vars(parameters) # namespace object
    
    type = parameters.get("type", "unknown")
    if type == "importance1":
        convert_importance(checkpoint, parameters, output_filename)
    elif type == "sparse1":
        convert_sparse(checkpoint, parameters, output_filename)
    elif type == "adaptive1" or type == "adaptive2":
        convert_adaptive(checkpoint, parameters, output_filename)
    elif type == "dense1" or type == "dense2": #new style dense super-resolution
        mode = parameters.get('mode', 'iso')
        if mode == 'iso':
            convert_unshaded(checkpoint, parameters, output_filename)
        elif mode == 'dvr':
            convert_dense_dvr(checkpoint, parameters, output_filename)
        else:
            raise ValueError("Unknown dense super-resolution mode: "+mode)
    elif type == "stepsize1":
        convert_stepsize(checkpoint, parameters, output_filename)
    else:
        # old versions
        if 'minImportance' in parameters:
            convert_importance(checkpoint, parameters, output_filename)
        elif 'upscale_factor' in parameters or 'upscaleFactor' in parameters:
            convert_unshaded(checkpoint, parameters, output_filename)
        else:
            convert_sparse(checkpoint, parameters, output_filename)

    print("Done")

if __name__== "__main__":
    parser = argparse.ArgumentParser(
        description="Converts PyTorch checkpoints from the training into executable models")
    parser.add_argument("input", help="Input file, a .pth file", nargs='?', default=None)
    parser.add_argument("output", help="Output file, a .pt file", nargs='?', default=None)
    args = parser.parse_args()

    folderMode = 0

    Tk().withdraw()
    if args.input is None:
        file = filedialog.askopenfilename(
            title="Select input checkpoint",
            filetypes = (("PyTorch Checkpoint", "*.pth"), ) )
        if file is None or not os.path.exists(file):
            print("No input file selected")
            exit(-1)
        input_filename = file
    else:
        if os.path.isdir(args.input):
            folderMode += 1
        elif not args.input.endswith(".pth"):
            print("Selected input file is not a .pth file")
            exit(-1)
        input_filename = args.input
    print("Selected input file:", input_filename)

    if args.output is None:
        file = filedialog.asksaveasfilename(
            title="Select output script file",
            filetypes = (("PyTorch Script", "*.pt"), ) )
        if not file.endswith(".pt"):
            file = file + ".pt"
        output_filename = file
    else:
        if os.path.isdir(args.output):
            folderMode += 1
        elif not args.output.endswith(".pt"):
            print("Selected output file is not a .pt file")
            exit(-1)
        output_filename = args.output
    if not output_filename[-3:]=='.pt' and folderMode==0:
        output_filename = output_filename + ".pt"
    print("Selected output file:", output_filename)

    if folderMode == 2:
        print("Batch convert")
        files = [f for f in os.listdir(input_filename) if f.endswith(".pth")]
        for f in files:
            input = os.path.join(input_filename, f)
            output = os.path.join(output_filename, f[:-4]+".pt")
            convert(input, output)
    elif folderMode == 0:
        convert(input_filename, output_filename)
    else:
        print("Either specify input and output files or input and output directories")