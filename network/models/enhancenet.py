import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from models.partialconv2d import PartialConv2dPackedMask

# EnhanceNet: Single Image Super-Resolution through Automated Texture Synthesis - Sajjadi et.al.
# https://github.com/geonm/EnhanceNet-Tensorflow
class EnhanceNet(nn.Module):
    def __init__(self, upscale_factor, input_channels, channel_mask, output_channels, opt):
        '''
        Additional options:
        upsample: nearest, bilinear, or pixelShuffler
        recon_type: residual or direct
        use_bn: for batch normalization of the residual blocks
        num_layers: the number of layers in the network, default=10
        '''
        super(EnhanceNet, self).__init__()
        #assert(upscale_factor==4)
        self.upscale_factor = upscale_factor
        self.upsample = opt.upsample if hasattr(opt, "upsample") else "bilinear"
        self.recon_type = opt.reconType if hasattr(opt, "reconType") else "residual"
        self.use_bn = opt.useBN if hasattr(opt, "useBN") else False
        self.channel_mask = channel_mask
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.return_residual = opt.return_residual if hasattr(opt, 'return_residual') else True
        self.num_layers = opt.num_layers if hasattr(opt, 'num_layers') else 10
        self.num_channels = opt.num_channels if hasattr(opt, 'num_channels') else 64
        self.padding = opt.padding if hasattr(opt, 'padding') else 'zero'
        if self.padding not in ['zero', 'partial']:
            raise ValueError("padding must be 'zero' or 'partial', but is '%s'"%self.padding)
        self.train_mask = opt.train_mask if hasattr(opt, 'train_mask') else False

        if self.padding == 'partial' and self.use_bn:
            raise ValueError("Partial padding and batch normalization not compatible yet.")
        # since with partial padding, the mask is added as extra channel, this extra channel
        # should be ignored in the batch normalization, but isn't at the moment.
        # This leads to: Runtime Error, running mean should contain 64 elements, but got 65

        self._enhancenet(input_channels, output_channels)
        self._initialize_weights()

    def _preprocess(self, images):
        #pp_images = images / 255.0
        ## simple mean shift
        images = images * 2.0 - 1.0

        return images
    
    def _postprocess(self, images):
        pp_images = ((images + 1.0) / 2.0)# * 255.0
        
        return pp_images

    def _upsample(self, factor):
        factor = float(factor)
        if self.upsample == 'nearest':
            return nn.Upsample(scale_factor=factor, mode='nearest')
        elif self.upsample == 'bilinear':
            return nn.Upsample(scale_factor=factor, mode='bilinear')
        elif self.upsample == 'bicubic':
            return nn.Upsample(scale_factor=factor, mode='bicubic')
        else: #pixelShuffle
            return nn.PixelShuffle(self.factor)

    #@profile
    def _recon_image(self, inputs, outputs):
        '''
        LR to HR -> inputs: LR, outputs: HR
        HR to LR -> inputs: HR, outputs: LR
        '''

        # check if we have recovered that model from a checkpoint prior to unshaded networks
        if not hasattr(self, 'channel_mask'):
            self.channel_mask = [0, 1, 2] # fallback to default RGB mode
        if not hasattr(self, 'output_channels'):
            self.output_channels = 3 # fallback to default RGB mode

        #resized_inputs = F.interpolate(inputs[:,self.channel_mask,:,:], 
        #inputs_masked = inputs[:,self.channel_mask,:,:]  #time per hit: 58463.9
        channel_mask_length = len(self.channel_mask)
        inputs_masked = inputs[:,0:channel_mask_length,:,:] #time per hit: 289.8
        resized_inputs = F.interpolate(inputs_masked, 
                                       size=[outputs.shape[2], 
                                             outputs.shape[3]], 
                                       mode=self.upsample)
        if self.recon_type == 'residual':
            if channel_mask_length==self.output_channels:
                recon_outputs = resized_inputs + outputs
            elif channel_mask_length<self.output_channels:
                recon_outputs = torch.cat([
                    resized_inputs + outputs[:,0:len(self.channel_mask),:,:],
                    outputs[:,len(self.channel_mask):,:,:]],
                    dim=1)
            else:
                raise ValueError("number of output channels must be at least the number of masked input channels")
        else:
            recon_outputs = outputs
        
        #resized_inputs = self._postprocess(resized_inputs)
        #resized_inputs = tf.cast(tf.clip_by_value(resized_inputs, 0, 255), tf.uint8)
        #tf.summary.image('4_bicubic image', resized_inputs)

        #recon_outputs = self._postprocess(recon_outputs)
        
        return recon_outputs, outputs
        
    def _enhancenet(self, input_channels, output_channels):
        num_channels = self.num_channels if hasattr(self, "num_channels") else 64

        def createConv2d(cin, cout, size=3, padding=1):
            if self.padding == 'partial':
                return PartialConv2dPackedMask(cin, cout, size, padding=padding, train_mask = self.train_mask)
            else:
                return nn.Conv2d(cin, cout, size, padding=padding)

        self.preblock = nn.Sequential(
            createConv2d(input_channels, num_channels, 3, padding=1),
            nn.ReLU())
            
        self.blocks = []
        for idx in range(self.num_layers):
            if self.use_bn:
                self.blocks.append(nn.Sequential(
                    createConv2d(num_channels, num_channels, 3, padding=1),
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(),
                    createConv2d(num_channels, num_channels, 3, padding=1),
                    nn.BatchNorm2d(num_channels)
                    ))
            else:
                self.blocks.append(nn.Sequential(
                    createConv2d(num_channels, num_channels, 3, padding=1),
                    nn.ReLU(),
                    createConv2d(num_channels, num_channels, 3, padding=1),
                    ))
        self.blocks = nn.ModuleList(self.blocks)
            
        postblock_entries = []
        factor = self.upscale_factor
        while factor >= 2:
            postblock_entries += [
                self._upsample(2),
                createConv2d(num_channels, num_channels, 3, padding=1),
                nn.ReLU()]
            factor /= 2
        if factor > 1:
            postblock_entries += [
                self._upsample(factor),
                createConv2d(num_channels, num_channels, 3, padding=1),
                nn.ReLU()]
        postblock_entries += [
            createConv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(),
            createConv2d(num_channels, output_channels, 3, padding=1)
            ]
        self.postblock = nn.Sequential(*postblock_entries)

    def _initialize_weights(self):
        def init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.orthogonal_(m.weight, torch.nn.init.calculate_gain('relu'))
        for block in self.blocks:
            block.apply(init)      

    #@profile
    def forward(self, input, mask = None):
        #inputs = self._preprocess(inputs)

        if self.padding == 'partial':
            if mask is None:
                mask = torch.ones((input.shape[0], 1, input.shape[2], input.shape[3]),
                                  dtype=input.dtype, device=input.device)
            inputWithMash = torch.cat([input, mask], dim=1)
        else:
            inputWithMash = input

        features = self.preblock(inputWithMash)
        for block in self.blocks:
            features = features + block(features)
        output = self.postblock(features)

        if self.padding == 'partial':
            output = output[:, :-1, :, :]

        output, residual = self._recon_image(input, output)
        if not hasattr(self, 'return_residual') or self.return_residual:
            return output, residual
        else:
            return output