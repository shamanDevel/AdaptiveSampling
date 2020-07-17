import torch
import torch.nn as nn
import torch.nn.functional as F

# TecoGAN generator
class TecoGAN(nn.Module):
    def __init__(self, upscale_factor, input_channels, channel_mask, output_channels, opt):
        '''
        upscaling_factor: how much to upscale, e.g. 4
        upsample: nearest, bilinear, or pixelShuffler
        recon_type: residual or direct
        use_bn: for batch normalization of the residual blocks
        '''
        super(TecoGAN, self).__init__()
        assert(upscale_factor==4)
        self.upscale_factor = upscale_factor
        self.upsample = opt.upsample
        self.recon_type = opt.reconType
        self.num_residual_layers = opt.numResidualLayers
        self.channel_mask = channel_mask

        self._enhancenet(input_channels, output_channels)
        #self._initialize_weights()

    def _recon_image(self, inputs, outputs):
        '''
        LR to HR -> inputs: LR, outputs: HR
        HR to LR -> inputs: HR, outputs: LR
        '''
        resized_inputs = F.interpolate(inputs[:,self.channel_mask,:,:], 
                                       size=[outputs.shape[2], 
                                             outputs.shape[3]], 
                                       mode=self.upsample)
        if self.recon_type == 'residual':
            recon_outputs = resized_inputs + outputs
        else:
            recon_outputs = outputs
        
        return recon_outputs, outputs
        
    def _enhancenet(self, input_channels, output_channels):
        self.preblock = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.LeakyReLU())
            
        self.blocks = []
        for idx in range(self.num_residual_layers):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                ))
        self.blocks = nn.ModuleList(self.blocks)
            
        self.postblock = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, output_channels, 3, padding=1),
                nn.LeakyReLU(),
            )

    def _initialize_weights(self):
        def init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.orthogonal_(m.weight, torch.nn.init.calculate_gain('relu'))
        for block in self.blocks:
            block.apply(init)      

    def forward(self, inputs):
        #inputs = self._preprocess(inputs)

        features = self.preblock(inputs)
        for block in self.blocks:
            features = features + block(features)
        outputs = self.postblock(features)

        outputs, residual = self._recon_image(inputs, outputs)
        return outputs, residual