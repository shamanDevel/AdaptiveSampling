"""
Generator networks.
Possible networks: EnhanceNet, SubpixelNet, TecoGAN.

Each net takes two arguments:
 - upscale_factor, currently always expected to be 4
 - opt: Namespace object of options: (if supported by the network)
    - upsample: upsample mode, can be 'nearest', 'bilinear' or 'pixelShuffler'
    - reconType: if 'residual', the whole network is a residual network
    - useBn: True if batch normalization should be used
    - numResidualLayers: integer specifying the number of residual layers

"""

from .enhancenet import EnhanceNet
from .subpixelnet import SubpixelNet
from .tecogan import TecoGAN
from .rcan import RCAN
from .videotools import VideoTools
from .unet import UNet

import dataset.datasetUtils
import torch

def createNetwork(name, upscale_factor, input_channels, channel_mask, output_channels, additional_opt):
    print('upscale_factor:', upscale_factor)
    print('input_channels:', input_channels)
    print('channel_mask:', channel_mask)
    print('output_channels:', output_channels)
    """
    Creates the network for single image superresolution.
    Parameters:
     - upscale_factor: the upscale factor of the network, assumed to be 4
     - input_channels: the number of input channels of the low resolution image.
        Can vary from setting to setting to include rgb, depth, normal, warped previous frames
     - channel_mask: selection of input channels that match the output channels.
        Used for the residual architecture
     - output_chanels: the number of ouput channels of the high resolution image.
        Can vary from setting to setting
     - additional_opt: additional command line parameters to the networks
    """
    model = None
    if name.lower()=='SubpixelNet'.lower():
        model = SubpixelNet(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    elif name.lower()=='EnhanceNet'.lower():
        model = EnhanceNet(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    elif name.lower()=='TecoGAN'.lower():
        model = TecoGAN(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    elif name.lower()=='RCAN'.lower():
        model = RCAN(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    else:
        raise ValueError('Unknown model %s'%name)
    return model


class NormalizedAndColorConvertedNetwork(torch.nn.Module):
    """
    wraps a network with channel normalization and color space conversion.
    The network takes a tuple as input, performs the conversion only on the first tensor.
    The network may also output a tuple of tensors, the de-conversion is also only done on the first tensor.
    """

    def __init__(self, network : torch.nn.Module, *,
                 normalize : dataset.datasetUtils.Normalization.Normalize = None,
                 denormalize : dataset.datasetUtils.Normalization.Denormalize = None,
                 colorSpace : str = None):
        """
        Applies normalization if 'normalize' and 'denormalize' is not None.
        Applies color space transformation afterwards if 'colorSpace' is not None.
        """
        super().__init__()
        from utils import colorConversion as cc
        self._network = network
        self._normalize = normalize
        self._denormalize = denormalize
        self._colorToSpace = cc.DvrColorToSpace(colorSpace != None, colorSpace or 'rgb')
        self._colorFromSpace = cc.DvrColorFromSpace(colorSpace != None, colorSpace or 'rgb')

    def forward(self, *args):
        args = list(args)
        assert len(args)>0

        if self._normalize is not None and self._denormalize is not None:
            args[0] = self._normalize(args[0])
        args[0] = self._colorToSpace(args[0])

        args = tuple(args)
        results = self._network(*args)
        if isinstance(results, tuple):
            results = list(results)
        else:
            results = [results]

        results[0] = self._colorFromSpace(results[0])
        if self._normalize is not None and self._denormalize is not None:
            results[0] = self._denormalize(results[0])

        if len(results)==1:
            return results[0]
        else:
            return tuple(results)
