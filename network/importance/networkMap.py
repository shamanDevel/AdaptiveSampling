import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import List

from importance.importanceMap import ImportanceMap  # relativ import
from importance.gradientMap import GradientImportanceMap
from models.enhancenet import EnhanceNet # import from global

class NetworkImportanceMap(ImportanceMap):
    """
    An importance Map that uses a trainable network to compute the heat map
    """

    def __init__(self, upsamplingFactor:float,
                 input_channels:int,
                 model:str = "EnhanceNet",
                 use_bn:bool = False,
                 border_padding:int = 8,
                 output_layer:str = 'none',
                 num_layers:int = 10,
                 residual:str = 'off',
                 residual_channels:List[int] = None,
                 padding:str = 'zero'):
        """
        Creates the Importance Map implementation using a neural network.
        So far only the EnhanceNet is supported (model='EnhanceNet').

        EnhanceNet parameters:
         input_channels: The expected input channels of the importance map computation
         use_bn:         use batch normalizations
         border_padding: zero-border padding of the input to avoid border effects
         output_layer:   The output activation layer. Can be
                         'none', 'softplus', 'sigmoid'.
                         Default: 'none'
        """
        super().__init__(upsamplingFactor)
        assert model=="EnhanceNet", "Only EnhanceNet is supported at the moment"

        assert residual=='off' or residual=='gradient', "residual can be either 'off' or 'gradient'"
        self._residual = residual
        if residual=='gradient':
            assert residual_channels is not None, "if residual='gradient', residual_channels must be specified"
            self._residual_net = GradientImportanceMap(upsamplingFactor, *[(c, 1) for c in residual_channels])

        #create the model
        self._input_channels = input_channels
        Opt = namedtuple("Opt", ['upsample', 'reconType', 'useBN', 'return_residual', 'num_layers', 'padding'])
        opt = Opt(upsample='bilinear', reconType='direct', useBN=use_bn, 
                  return_residual=False, num_layers=num_layers, padding=padding)
        self._model = EnhanceNet(
            upscale_factor = upsamplingFactor,
            input_channels = input_channels,
            channel_mask = list(range(input_channels)),
            output_channels = 1,
            opt = opt)
        if output_layer == 'softplus':
            self._model = nn.Sequential(
                self._model,
                nn.Softplus())
        elif output_layer == 'sigmoid':
            self._model = nn.Sequential(
                self._model,
                nn.Sigmoid())
        self._border_padding = border_padding

    def network(self):
        """Returns the network"""
        return self._model

    def forward(self, input):
        shape = input.shape
        assert shape[1] == self._input_channels, \
            "input channels from the constructor don't match current channel count, expected %d, got %d"%(
                self._input_channels, shape[1])
        # pad input
        input = F.pad(input, [self._border_padding]*4, 'constant', 0)
        # run network
        output = self._model(input)
        # add residual
        residual_mode = self._residual if hasattr(self, '_residual') else 'off'
        if residual_mode == 'gradient':
            output = output + self._residual_net(input)
        # remove padding
        pad = self._border_padding * self._upsampleFactor
        return F.pad(output, [-pad]*4, 'constant', 0)
        #return output[:,0,pad:-pad-1, pad:-pad-1]

