import torch
import torch.nn as nn
import torch.nn.functional as F

class RCAN(nn.Module):
    """
    RCAN: 
    Image Super-Resolution Using Very Deep Residual Channel Attention Networks,
    Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu
    """

    def __init__(self, upscale_factor, input_channels, channel_mask, output_channels, opt):
        super().__init__()
        assert(upscale_factor==4)
        self.upscale_factor = upscale_factor

        # TODO: read parameters from the config
        self.upsample = 'pixelShuffle' # nearest, bilinear or pixelShuffle, as in the paper
        self.num_blocks_g = 10 # G: number of outer RG blocks, as in the paper
        self.num_blocks_b = 20 # B: number of innter RCAB blocks per RG block, as in the paper
        self.num_channels = 64 # C: number of channels in the convolutions
        self.channel_downscaling = 16 # r: channel downscaling for the CA blocks
        self.reduced_channels = self.num_channels // self.channel_downscaling
        self.channel_mask = channel_mask

        self._build_net(input_channels, output_channels)

    class ChannelAttentionBlock(nn.Module):
        def __init__(self, rcan):
            super().__init__()
            #self.downscaling = nn.Conv1d(rcan.num_channels, rcan.reduced_channels, 1)
            #self.upscaling = nn.Conv1d(rcan.reduced_channels, rcan.num_channels, 1)
            self.downscaling = nn.Linear(rcan.num_channels, rcan.reduced_channels)
            self.upscaling = nn.Linear(rcan.reduced_channels, rcan.num_channels)
        def forward(self, x):
            b,c,w,h = x.shape
            z = torch.mean(x.view(b,c,w*h), dim=2)
            s = self.downscaling(z)
            s = F.leaky_relu(s)
            s = self.upscaling(s)
            s = torch.sigmoid(s)
            s = s.view(b, c, 1, 1)
            return x * s
    def _ca(self): # Channel Attention (CA) block
        return RCAN.ChannelAttentionBlock(self)

    class ResidualChannelAttentionBlock(nn.Module):
        def __init__(self, rcan):
            super().__init__()
            self.pre = nn.Sequential(
                nn.Conv2d(rcan.num_channels, rcan.num_channels, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(rcan.num_channels, rcan.num_channels, 3, padding=1)
                )
            self.ca = rcan._ca()
        def forward(self, x):
            return x + self.ca(self.pre(x))
    def _rcab(self): # Residual Channel Attention (RCAB) block
        return RCAN.ResidualChannelAttentionBlock(self)

    class ResGroup(nn.Module):
        def __init__(self, rcan):
            super().__init__()
            self.blocks = nn.ModuleList([rcan._rcab() for i in range(rcan.num_blocks_b)])
            self.post = nn.Conv2d(rcan.num_channels, rcan.num_channels, 3, padding=1)
        def forward(self, x):
            f = x
            for block in self.blocks:
                f = block(f)
            f = self.post(f)
            return f + x
    def _resgroup(self): # Residual group of B RCABs with short skip connections
        return RCAN.ResGroup(self)

    class RIR(nn.Module):
        def __init__(self, rcan):
            super().__init__()
            self.blocks = nn.ModuleList([rcan._resgroup() for i in range(rcan.num_blocks_g)])
            self.post = nn.Conv2d(rcan.num_channels, rcan.num_channels, 3, padding=1)
        def forward(self, x):
            f = x
            for block in self.blocks:
                f = block(f)
            f = self.post(f)
            return f + x
    def _rir(self): # residual in residual blocks wih long skip connection
        return RCAN.RIR(self)

    def _upsample(self, factor=None, in_channels=None):
        if factor is None:
            factor = self.upscale_factor
        if in_channels is None:
            in_channels = self.num_channels
        if self.upsample == 'nearest':
            return nn.UpsamplingNearest2d(scale_factor=factor), in_channels
        elif self.upsample == 'bilinear':
            return nn.UpsamplingBilinear2d(scale_factor=factor), in_channels
        elif self.upsample == 'pixelShuffle':
            return nn.PixelShuffle(factor), in_channels // (factor**2)
        else:
            raise ValueError('Unknown upsample mode %s'%self.upsample)

    def _build_net(self, input_channels, output_channels):
        upsample, upsample_channels = self._upsample()
        self.net = nn.ModuleDict({
            'pre':nn.Conv2d(input_channels, self.num_channels, 3, padding=1),
            'rir':self._rir(),
            'up':upsample,
            'post':nn.Conv2d(upsample_channels, output_channels, 3, padding=1)
            })

    def forward(self, inputs):
        x = self.net['pre'](inputs)
        x = self.net['rir'](x)
        x = self.net['up'](x)
        outputs = self.net['post'](x)
        residual = outputs - F.interpolate(inputs[:,self.channel_mask,:,:], 
                                           size=(outputs.shape[2], outputs.shape[3]),
                                           mode='bilinear')
        outputs = torch.clamp(outputs, 0, 1)
        return outputs, residual


