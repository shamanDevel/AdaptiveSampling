"""
These utilities perform color space conversions.
They always convert from RGB to a user-specified color space
"""

import torch
import torch.nn as nn

try:
    import kornia.color as kc
except ImportError:
    print("Unable to find kornia, color conversions won't work")

class Rgb2Space(nn.Module):
    def __init__(self, target_space : str):
        super().__init__()
        self._target_space = target_space

    def forward(self, x):
        if self._target_space == 'rgb':
            return x
        elif self._target_space == 'hsv':
            output = kc.rgb_to_hsv(x)
            #print("RGB in range ({}, {}) to HSV in range ({},{})".format(
            #    x.min().item(), x.max().item(), output.min().item(), output.max().item()))
            return output
        elif self._target_space == 'luv':
            return kc.rgb_to_luv(x)
        elif self._target_space == 'xyz':
            return kc.rgb_to_xyz(x)
        else:
            raise ValueError("this should not happen")


class Space2Rgb(nn.Module):
    def __init__(self, target_space : str):
        super().__init__()
        self._target_space = target_space

    def forward(self, x):
        if self._target_space == 'rgb':
            return x
        elif self._target_space == 'hsv':
            output = kc.hsv_to_rgb(x)
            #print("HSV in range ({}, {}) to RGB in range ({},{})".format(
            #    x.min().item(), x.max().item(), output.min().item(), output.max().item()))
            return output
        elif self._target_space == 'luv':
            return kc.luv_to_rgb(x)
        elif self._target_space == 'xyz':
            return kc.xyz_to_rgb(x)
        else:
            raise ValueError("this should not happen")


def getColorSpaceConversions(target_space : str):
    """
    For a given target colorspace (RGB, HSV, case-insensitive),
    returns a tuple with modules for
     - converting to that space from RGB
     - converting back to RGB
    """
    target_space = target_space.lower()
    assert target_space in ['rgb', 'hsv', 'xyz', 'luv'], \
        "only RGB, HSV, XYZ and LUV are supported, not %s"%target_space
    return Rgb2Space(target_space), Space2Rgb(target_space)


class DvrColorToSpace(nn.Module):
    """
    Application of color-space convertion to DVR images, if enabled.
    This transforms the first three channels
    """

    def __init__(self, enable : bool, space : str):
        super().__init__()
        self._enable = enable
        self._space = space
        self._colorConvertToSpace = getColorSpaceConversions(space)[0]

    def forward(self, input : torch.Tensor):
        if self._enable:
            return torch.cat((
                    self._colorConvertToSpace(input[..., 0:3, :, :]),
                    input[..., 3:, :, :]),
                    dim=-3)
        else:
            return input

class DvrColorFromSpace(nn.Module):
    """
    Application of color-space convertion to DVR images, if enabled.
    This transforms the first three channels
    """

    def __init__(self, enable : bool, space : str):
        super().__init__()
        self._enable = enable
        self._space = space
        self._colorConvertFromSpace = getColorSpaceConversions(space)[1]

    def forward(self, input : torch.Tensor):
        if self._enable:
            return torch.cat((
                    self._colorConvertFromSpace(input[..., 0:3, :, :]),
                    input[..., 3:, :, :]),
                    dim=-3)
        else:
            return input