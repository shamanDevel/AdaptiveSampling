# Source: https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F
from .partialconv2d import PartialConv2d

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        depth=5,
        wf=6,
        padding='zero',
        batch_norm=False,
        residual=False,
        hard_input=False,
        up_mode='upconv',
        return_masks=False
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (str): one of 'off', 'zero' (default) or 'partial'.
                           'off' will not use any padding, the output size
                             will differ from the input size
                           'zero' will use a padding of 1, the output size
                             will match the input size
                           'partial' will use partial convolutions, the output
                             size will match the input size.
                             This assumes that a mask is provided
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            residual (bool): if True, the input channels are added
                             to the output channels.
                             If in_channels!=out_channels, only the first
                             min(in_channels, out_channels) are used.
            hard_input (bool): if True, the pixels from the input
                               with mask=1 are directly copied to the output
                               and overwrites whatever the network produces.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            return_masks (bool): If True, the list of masks is returned
                                 together with the output.
                                 If padding!='partial', the returned list
                                 will be empty
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if padding=='partial':
                self.down_path.append(
                    UNetPartialConvBlock(prev_channels, 2 ** (wf + i), batch_norm)
                )
            else:
                self.down_path.append(
                    UNetNormalConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
                )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        
        self.residual = residual
        self.hard_input = hard_input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.return_masks = return_masks

    def forward(self, input, mask=None):
        """
        Forward pass,
        the mask has to be defined if partial convolutions are used.
        The mask is a Bx1xHxW float tensor of either 1 (data valid) or 0
        """

        """
        Implementation Note:
        For partial convolutions, the mask is passed around as the first channel.
        This channel has to be stripped away 
        """

        blocks = []
        masks = []

        #for partial convolutions, add the mask as first channel
        if self.padding == 'partial':
            x = torch.cat([input, mask], dim=1)
        else:
            x = input

        # downward
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        #for partial convolutions, strip away the mask
        if self.padding == 'partial':
            masks.append(x[:,0:1,:,:])
            x = x[:,1:,:,:]
            for i in range(len(blocks)):
                masks.append(blocks[-i-1][:,0:1,:,:])
                blocks[-i-1] = blocks[-i-1][:,1:,:,:]

        #upward
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        #convolution to the desired output channels
        x = self.last(x)

        # residual connections from input to output
        if self.residual:
            if self.in_channels < self.out_channels:
                x = torch.cat([
                    x[:,0:self.in_channels,:,:] + input,
                    x[:,self.in_channels:,:,:]],
                    dim=1)
            elif self.in_channels > self.out_channels:
                x = x + input[:,0:self.out_channels,:,:]
            else:
                x = x + input

        # hard input
        if self.hard_input:
            assert mask is not None, "hard_input requires the mask to be passed on"
            assert mask.shape[1] == 1

            # HACK: copy everything except AO (channel 5)
            assert self.in_channels == 9 + 8
            assert self.out_channels == 8
            x = torch.cat([
                x[:,0:5,:,:] * (1-mask) + input[:,0:5,:,:] * mask,
                x[:,5:6,:,:], # AO
                x[:,6:8,:,:] * (1-mask) + input[:,6:8,:,:] * mask],
                dim=1)

            #if self.in_channels < self.out_channels:
            #    x = torch.cat([
            #        x[:,0:self.in_channels,:,:] * (1-mask) + input * mask,
            #        x[:,self.in_channels:,:,:]],
            #        dim=1)
            #elif self.in_channels > self.out_channels:
            #    x = x * (1-mask) + input[:,0:self.out_channels,:,:] * mask
            #else:
            #    x = x * (1-mask) + input * mask

        #print("Input size:", input.shape, ", output size:", x.shape)

        if self.return_masks:
            return x, tuple(masks)
        else:
            return x

class UNetNormalConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []

        if padding=='off':
            pad = 0
        elif padding=='zero':
            pad = 1
        else:
            raise ValueError("padding must be 'off', 'zero' or 'partial'")

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=pad))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=pad))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        #print("UNetNormalConvBlock:", x.shape,'->',out.shape)
        return out

class UNetPartialConvBlock(nn.Module):
    """
    The mask is encoded in the first channel
    """
    def __init__(self, in_size, out_size, batch_norm):
        super().__init__()
        
        self.conv1 = PartialConv2d(in_size, out_size, kernel_size=3, padding=1, 
                                   return_mask = True)
        post1 = []
        post1.append(nn.ReLU())
        if batch_norm:
            post1.append(nn.BatchNorm2d(out_size))
        self.post1 = nn.Sequential(*post1)

        self.conv2 = PartialConv2d(out_size, out_size, kernel_size=3, padding=1, 
                                   return_mask = True)
        post2 = []
        post2.append(nn.ReLU())
        if batch_norm:
            post2.append(nn.BatchNorm2d(out_size))
        self.post2 = nn.Sequential(*post2)

    def forward(self, x):
        # extract mask
        mask = x[:,0:1,:,:]
        x = x[:,1:,:,:]

        # run block
        out, mask = self.conv1(x, mask)
        out = self.post1(out)
        out, mask = self.conv2(out, mask)
        out = self.post2(out)

        # merge mask
        out = torch.cat([mask, out], dim=1)
        #print("UNetPartialConvBlock:", x.shape,'->',out.shape)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        padding = "zero" if padding=="partial" else padding
        self.conv_block = UNetNormalConvBlock(in_size, out_size, padding, batch_norm)

    def center_pad(self, layer, target_size):
        """
        pads 'layer' so that it matches the 'target_size'.
        """
        # TODO: replace by F.pad with negative padding size
        # or negative indices
        # to avoid the size-dependent constants
        # They don't work with TorchScript..
        _, _, layer_height, layer_width = layer.size()
        diff_y_left = (layer_height - target_size[0]) // 2
        diff_y_right = layer_height - target_size[0] - diff_y_left
        diff_x_left = (layer_width - target_size[1]) // 2
        diff_x_right = layer_width - target_size[1] - diff_x_left
        pad = (-diff_x_left, -diff_x_right, -diff_y_left, -diff_y_right)
        #print("center_crop: current=(", layer_height, ", ", layer_width,
        #      "), target=(", target_size[0], ", ", target_size[1],
        #      ") -> pad=", pad,
        #      sep='')
        return F.pad(layer, pad)
        #return layer[:,:, diff_y:-diff_y, diff_x:-diff_x]
        #return layer[
        #    :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        #]

    def forward(self, x, bridge):
        up = self.up(x)
        #crop1 = self.center_pad(bridge, up.shape[2:])
        #out = torch.cat([up, crop1], 1)
        crop2 = self.center_pad(up, bridge.shape[2:])
        out = torch.cat([crop2, bridge], 1)
        out = self.conv_block(out)
        #print("UNetUpBlock:", x.shape,'->',out.shape)
        return out