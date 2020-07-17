###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
# Source: https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class _dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Constructor for partial 2d convolutions.
        Additional keyword arguments:

        - multi_channel: True if the mask is multi-channel (B,C,H,W) or not (B,1,H,W).
            default: False
        - return_mask: if True, forward returns a tuple (output, mask), else just the output
            default: False
        - train_mask: if True, the mask is trainable and gradients are propagated to it
            default: False
        """


        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        if 'train_mask' in kwargs:
            self.train_mask = kwargs['train_mask']
            kwargs.pop('train_mask')
        else:
            self.train_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            ctx = _dummy_context_mgr() if self.train_mask else torch.no_grad()
            with ctx:
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class PartialConv2dPackedMask(PartialConv2d):
    def __init__(self, *args, **kwargs):
        """
        Constructor for partial 2d convolutions where the mask is passed as the last
        channel in the input and output. This channel is not passed to the regular convolution.

        This implies multi_channel=False and return_mask=True of PartialConv2d.
        Additional keyword arguments:
        - train_mask: if True, the mask is trainable and gradients are propagated to it
            default: False
        """
        kwargs['multi_channel']=False
        kwargs['return_mask']=True
        super(PartialConv2dPackedMask, self).__init__(*args, **kwargs)

    def forward(self, input):
        x = input[:, :-1, :, :].contiguous()
        m = input[:, -1:, :, :].contiguous()
        ox, om = super(PartialConv2dPackedMask, self).forward(x, m)
        return torch.cat([ox, om], dim=1)