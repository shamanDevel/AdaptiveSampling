import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lossbuilder import LossBuilder
from utils import ScreenSpaceShading

class LossNetUnshaded(nn.Module):
    """
    Main Loss Module for unshaded data.

    device: cpu or cuda
    losses: list of loss terms with weighting as string
       Format: <loss>:<target>:<weighting>
       with: loss in {l1, l2, perceptual, gan}
             target in {mask, normal, color, ao, all}, 'all' is only allowed for GAN
             weighting a positive number
    opt: command line arguments (Namespace object) with
     further parameters depending on the losses

    """
    def __init__(self, device, input_channels, output_channels, high_res, padding, losses, opt):
        super().__init__()
        self.padding = padding
        assert input_channels == 5 # mask, normalX, normalY, normalZ, depth
        assert output_channels == 6 # mask, normalX, normalY, normalZ, depth, ambient occlusion
        #List of tuples (name, weight or None)
        self.loss_list = [s.split(':') for s in losses.split(',')]
        
        # Build losses and weights
        builder = LossBuilder(device)
        self.loss_dict = {}
        self.weight_dict = {}

        self.loss_dict['mse:color'] = builder.mse() #always use mse for psnr
        self.weight_dict[('mse','color')] = 0.0

        content_layers = []
        style_layers = []
        self.has_discriminator = False
        self.has_style_or_content_loss = False
        self.has_temporal_l2_loss = False

        TARGET_INFO = {
            # target name : (#channels, expected min, expected max)
            "mask" : (1, -1, +1),
            "normal" : (3, -1, +1),
            "color" : (3, 0, 1),
            "ao" : (1, 0, 1),
            "depth" : (1, 0, 1)
            }

        for entry in self.loss_list:
            if len(entry)<2:
                raise ValueError("illegal format for loss list: " + entry)
            name = entry[0]
            target = entry[1]
            weight = entry[2] if len(entry)>2 else None
            if target!='mask' and target!='normal' and target!='color' and target!='ao' and target!='depth' and target!='all':
                raise ValueError("Unknown target: " + target)

            if 'mse'==name or 'l2'==name or 'l2_loss'==name:
                self.weight_dict['mse:'+target] = float(weight) if weight is not None else 1.0
            elif 'l1'==name or 'l1_loss'==name:
                self.loss_dict['l1:'+target] = builder.l1_loss()
                self.weight_dict[('l1',target)] = float(weight) if weight is not None else 1.0
            elif 'tl2'==name or 'temp-l2'==name:
                self.loss_dict['temp-l2:'+target] = builder.mse()
                self.weight_dict[('temp-l2',target)] = float(weight) if weight is not None else 1.0
                self.has_temporal_l2_loss = True
            elif 'tl1'==name or 'temp-l1'==name:
                self.loss_dict['temp-l1'] = builder.l1_loss()
                self.weight_dict[('temp-l1',target)] = float(weight) if weight is not None else 1.0
                self.has_temporal_l2_loss = True
            elif 'l2-ds'==name:
                self.loss_dict['l2-ds:'+target] = builder.downsample_loss(
                    'l2', opt.upscale_factor, 'bilinear')
                self.weight_dict[('l2-ds',target)] = float(weight) if weight is not None else 1.0
            elif 'l1-ds'==name:
                self.loss_dict['l1-ds:'+target] = builder.downsample_loss(
                    'l1', opt.upscale_factor, 'bilinear')
                self.weight_dict[('l1-ds',target)] = float(weight) if weight is not None else 1.0
            elif 'perceptual'==name:
                content_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in opt.perceptualLossLayers.split(',')]
                self.weight_dict[('perceptual',target)] = float(weight) if weight is not None else 1.0
                self.has_style_or_content_loss = True
            elif 'texture'==name:
                style_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in opt.textureLossLayers.split(',')]
                #style_layers = [('conv_1',1), ('conv_3',1), ('conv_5',1)]
                self.weight_dict[('texture',target)] = float(weight) if weight is not None else 1.0
                self.has_style_or_content_loss = True
            elif 'adv'==name or 'gan'==name: #spatial-temporal adversary
                assert target=='all'
                self.discriminator, self.adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    26, #5+5+8+8
                    opt)
                self.weight_dict[('adv',target)] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = False
                self.has_discriminator = True
            elif 'tgan'==name: #temporal adversary, current high-res + previous high-res
                assert target=='all'
                self.temp_discriminator, self.temp_adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    8+8,
                    opt)
                self.weight_dict[('tgan',target)] = float(weight) if weight is not None else 1.0
                self.has_discriminator = True
            elif 'sgan'==name: #spatial adversary, current high-res + current input
                assert target=='all'
                self.spatial_discriminator, self.spatial_adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    5+8,
                    opt)
                self.weight_dict[('sgan',target)] = float(weight) if weight is not None else 1.0
                self.has_discriminator = True
            elif 'ssim'==name:
                self.loss_dict['ssim:'+target] = builder.ssim_loss(TARGET_INFO[target][0])
                self.weight_dict[('ssim',target)] = float(weight) if weight is not None else 1.0
            elif 'dssim'==name:
                self.loss_dict['dssim:'+target] = builder.dssim_loss(TARGET_INFO[target][0])
                self.weight_dict[('dssim',target)] = float(weight) if weight is not None else 1.0
            elif 'lpips'==name:
                self.loss_dict['lpips:'+target] = builder.lpips_loss(*TARGET_INFO[target])
                self.weight_dict[('lpips',target)] = float(weight) if weight is not None else 1.0
            else:
                raise ValueError('unknown loss %s'%name)

        if self.has_style_or_content_loss:
            self.pt_loss, self.style_losses, self.content_losses = \
                    builder.get_style_and_content_loss(dict(content_layers), dict(style_layers))

        self.loss_dict = nn.ModuleDict(self.loss_dict)
        print('Loss weights:', self.weight_dict)

        self.shading = ScreenSpaceShading(device)
        self.shading.fov(30)
        self.shading.ambient_light_color(np.array([opt.lossAmbient, opt.lossAmbient, opt.lossAmbient]))
        self.shading.diffuse_light_color(np.array([opt.lossDiffuse, opt.lossDiffuse, opt.lossDiffuse]))
        self.shading.specular_light_color(np.array([opt.lossSpecular, opt.lossSpecular, opt.lossSpecular]))
        self.shading.specular_exponent(16)
        self.shading.enable_specular = False
        self.shading.light_direction(np.array([0.0, 0.0, 1.0]))
        self.shading.material_color(np.array([1.0, 1.0, 1.0]))
        self.shading.ambient_occlusion(opt.lossAO)
        print('LossNet: ambient occlusion strength:', opt.lossAO)

    def get_discr_parameters(self):
        params = []
        if hasattr(self, 'discriminator'):
            params = params + list(self.discriminator.parameters())
        if hasattr(self, 'temp_discriminator'):
            params = params + list(self.temp_discriminator.parameters())
        if hasattr(self, 'spatial_discriminator'):
            params = params + list(self.spatial_discriminator.parameters())
        return params

    def discr_eval(self):
        if hasattr(self, 'discriminator'): self.discriminator.eval()
        if hasattr(self, 'temp_discriminator'): self.temp_discriminator.eval()
        if hasattr(self, 'spatial_discriminator'): self.spatial_discriminator.eval()

    def discr_train(self):
        if hasattr(self, 'discriminator'): self.discriminator.train()
        if hasattr(self, 'temp_discriminator'): self.temp_discriminator.train()
        if hasattr(self, 'spatial_discriminator'): self.spatial_discriminator.train()

    def print_summary(self, gt_shape, pred_shape, input_shape, prev_pred_warped_shape, num_batches, device):
        #Print networks for VGG + Discriminator
        import torchsummary
        res = gt_shape[1]
        if 'perceptual' in self.weight_dict.keys() or 'texture' in self.weight_dict.keys():
            print('VGG (Perceptual + Style loss)')
            torchsummary.summary(self.pt_loss, (3, res, res), 2*num_batches, device=device.type)
        if hasattr(self, 'discriminator'):
            print('Discriminator:')
            if self.discriminator_use_previous_image:
                #2x mask+normal+color+ao
                input_images_shape = (16, res, res)
            else:
                # mask+normal+color+ao
                input_images_shape = (8, res, res)
            torchsummary.summary(
                self.discriminator,
                input_images_shape, 
                2*num_batches,
                device=device.type)


    @staticmethod
    def pad(img, border):
        """
        overwrites the border of 'img' with zeros.
        The size of the border is specified by 'border'.
        The output size is not changed.
        """
        if border==0: 
            return img
        b,c,h,w = img.shape
        img_crop = img[:,:,border:h-border,border:h-border]
        img_pad = F.pad(img_crop, (border, border, border, border), 'constant', 0)
        _,_,h2,w2 = img_pad.shape
        assert(h==h2)
        assert(w==w2)
        return img_pad

    #@profile
    def forward(self, gt, pred, input, prev_input_warped, prev_pred_warped, use_checkpoints = False):
        """
        gt: ground truth high resolution image (B x C=output_channels x 4W x 4H)
        pred: predicted high resolution image (B x C=output_channels x 4W x 4H)
        input: upsampled low resolution input image (B x C=input_channels x 4W x 4H)
               Only used for the discriminator, can be None if only the other losses are used
        prev_input_warped: upsampled, warped previous input image
        prev_pred_warped: predicted image from the previous frame warped by the flow
               Shape: B x Cout x 4W x 4H
               Only used for temporal losses, can be None if only the other losses are used
        """

        B, Cout, Hhigh, Whigh = gt.shape
        assert Cout == 6
        assert gt.shape == pred.shape

        # zero border padding
        gt = LossNetUnshaded.pad(gt, self.padding)
        pred = LossNetUnshaded.pad(pred, self.padding)
        if prev_pred_warped is not None:
            prev_pred_warped = LossNetUnshaded.pad(prev_pred_warped, self.padding)

        # extract mask and normal
        gt_mask = gt[:,0:1,:,:]
        gt_mask_clamp = torch.clamp(gt_mask*0.5 + 0.5, 0, 1)
        gt_normal = ScreenSpaceShading.normalize(gt[:,1:4,:,:], dim=1)
        gt_depth = gt[:,4:5,:,:]
        gt_ao = gt[:,5:6,:,:]
        pred_mask = pred[:,0:1,:,:]
        pred_mask_clamp = torch.clamp(pred_mask*0.5 + 0.5, 0, 1)
        pred_normal = ScreenSpaceShading.normalize(pred[:,1:4,:,:], dim=1)
        pred_depth = pred[:,4:5,:,:]
        pred_ao = pred[:,5:6,:,:]
        input_mask = input[:,0:1,:,:]
        input_mask_clamp = torch.clamp(input_mask*0.5 + 0.5, 0, 1)
        input_normal = ScreenSpaceShading.normalize(input[:,1:4,:,:], dim=1)
        input_depth = input[:,4:5,:,:]
        input_ao = None #not available

        # compute color output
        gt_color = self.shading(gt)
        pred_color = self.shading(pred)
        input_color = self.shading(input)

        generator_loss = 0.0
        loss_values = {}

        # normal, simple losses, uses gt+pred
        for name in ['mse','l1','ssim', 'dssim', 'lpips']:
            if (name,'mask') in self.weight_dict.keys():
                loss = self.loss_dict[name+':mask'](gt_mask, pred_mask)
                loss_values[(name,'mask')] = loss.item()
                generator_loss += self.weight_dict[(name,'mask')] * loss
            if (name,'normal') in self.weight_dict.keys():
                loss = self.loss_dict[name+':normal'](gt_normal*gt_mask_clamp, pred_normal*gt_mask_clamp)
                loss_values[(name,'normal')] = loss.item()
                generator_loss += self.weight_dict[(name,'normal')] * loss
            if (name,'ao') in self.weight_dict.keys():
                loss = self.loss_dict[name+':ao'](gt_ao*gt_mask_clamp, pred_ao*gt_mask_clamp)
                loss_values[(name,'ao')] = loss.item()
                generator_loss += self.weight_dict[(name,'ao')] * loss
            if (name,'depth') in self.weight_dict.keys():
                loss = self.loss_dict[name+':depth'](gt_depth*gt_mask_clamp, pred_depth*gt_mask_clamp)
                loss_values[(name,'depth')] = loss.item()
                generator_loss += self.weight_dict[(name,'depth')] * loss
            if (name,'color') in self.weight_dict.keys():
                loss = self.loss_dict[name+':color'](gt_color, pred_color)
                loss_values[(name,'color')] = loss.item()
                generator_loss += self.weight_dict[(name,'color')] * loss

        # downsample loss, use input+pred
        # TODO: input is passed in upsampled version, but this introduces more errors
        # Better: input low-res input and use the 'low_res_gt=True' option in downsample_loss
        # This requires the input to be upsampled here for the GAN.
        for name in ['l2-ds', 'l1-ds']:
            if (name,'mask') in self.weight_dict.keys():
                loss = self.loss_dict[name+':mask'](input_mask, pred_mask)
                loss_values[(name,'mask')] = loss.item()
                generator_loss += self.weight_dict[(name,'mask')] * loss
            if (name,'normal') in self.weight_dict.keys():
                loss = self.loss_dict[name+':normal'](input_normal*input_mask_clamp, pred_normal*input_mask_clamp)
                loss_values[(name,'normal')] = loss.item()
                generator_loss += self.weight_dict[(name,'normal')] * loss
            if (name,'ao') in self.weight_dict.keys():
                loss = self.loss_dict[name+':ao'](input_ao*input_mask_clamp, pred_ao*input_mask_clamp)
                loss_values[(name,'ao')] = loss.item()
                generator_loss += self.weight_dict[(name,'ao')] * loss
            if (name,'depth') in self.weight_dict.keys():
                loss = self.loss_dict[name+':depth'](input_depth*input_mask_clamp, pred_depth*input_mask_clamp)
                loss_values[(name,'depth')] = loss.item()
                generator_loss += self.weight_dict[(name,'depth')] * loss
            if (name,'color') in self.weight_dict.keys():
                loss = self.loss_dict[name+':color'](input_color, pred_color)
                loss_values[(name,'color')] = loss.item()
                generator_loss += self.weight_dict[(name,'color')] * loss

        # special losses: perceptual+texture, uses gt+pred
        def compute_perceptual(target, in_pred, in_gt):
            if ('perceptual',target) in self.weight_dict.keys() \
                    or ('texture',target) in self.weight_dict.keys():
                style_weight=self.weight_dict.get(('texture',target), 0)
                content_weight=self.weight_dict.get(('perceptual',target), 0)
                style_score = 0
                content_score = 0

                input_images = torch.cat([in_gt, in_pred], dim=0)
                self.pt_loss(input_images)

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                if ('perceptual',target) in self.weight_dict.keys():
                    loss_values[('perceptual',target)] = content_score.item()
                if ('texture',target) in self.weight_dict.keys():
                    loss_values[('texture',target)] = style_score.item()
                return style_weight * style_score + content_weight * content_score
            return 0
        generator_loss += compute_perceptual('mask', pred_mask.expand(-1,3,-1,-1)*0.5+0.5, gt_mask.expand(-1,3,-1,-1)*0.5+0.5)
        generator_loss += compute_perceptual('normal', (pred_normal*gt_mask_clamp)*0.5+0.5, (gt_normal*gt_mask_clamp)*0.5+0.5)
        generator_loss += compute_perceptual('color', pred_color, gt_color)
        generator_loss += compute_perceptual('ao', pred_ao.expand(-1,3,-1,-1), gt_ao.expand(-1,3,-1,-1))
        generator_loss += compute_perceptual('depth', pred_depth.expand(-1,3,-1,-1), gt_depth.expand(-1,3,-1,-1))

        # special: discriminator, uses input+pred+prev_pred_warped
        if self.has_discriminator:
            pred_with_color = torch.cat([
                pred_mask,
                pred_normal,
                pred_color,
                pred_ao], dim=1)

            prev_pred_normal = ScreenSpaceShading.normalize(prev_pred_warped[:,1:4,:,:], dim=1)
            prev_pred_with_color = torch.cat([
                prev_pred_warped[:,0:1,:,:],
                prev_pred_normal,
                self.shading(torch.cat([
                    prev_pred_warped[:,0:1,:,:],
                    prev_pred_normal,
                    prev_pred_warped[:,4:6,:,:]
                    ], dim=1)),
                prev_pred_warped[:,5:6,:,:]
                ], dim=1)

            input_pad = LossNetUnshaded.pad(input, self.padding)
            prev_input_warped_pad = LossNetUnshaded.pad(prev_input_warped, self.padding)
            pred_with_color_pad = LossNetUnshaded.pad(pred_with_color, self.padding)
            prev_pred_warped_pad = LossNetUnshaded.pad(prev_pred_with_color, self.padding)

            if ('adv','all') in self.weight_dict.keys(): # spatial-temporal
                discr_input = torch.cat([input_pad, prev_input_warped_pad, pred_with_color_pad, prev_pred_warped_pad], dim=1)
                gen_loss = self.adv_loss(self.discriminator(discr_input))
                loss_values['discr_pred'] = gen_loss.item()
                generator_loss += self.weight_dict[('adv','all')] * gen_loss

            if ('tgan','all') in self.weight_dict.keys(): #temporal
                discr_input = torch.cat([pred_with_color_pad, prev_pred_warped_pad], dim=1)
                gen_loss = self.temp_adv_loss(self.temp_discriminator(discr_input))
                loss_values['temp_discr_pred'] = gen_loss.item()
                generator_loss += self.weight_dict[('tgan','all')] * gen_loss

            if ('sgan','all') in self.weight_dict.keys(): #spatial
                discr_input = torch.cat([input_pad, pred_with_color_pad], dim=1)
                gen_loss = self.spatial_adv_loss(self.spatial_discriminator(discr_input))
                loss_values['spatial_discr_pred'] = gen_loss.item()
                generator_loss += self.weight_dict[('sgan','all')] * gen_loss

        # special: temporal l2 loss, uses input (for the mask) + pred + prev_warped
        if self.has_temporal_l2_loss:
            prev_pred_mask = prev_pred_warped[:,0:1,:,:]
            prev_pred_normal = ScreenSpaceShading.normalize(prev_pred_warped[:,1:4,:,:], dim=1)
            for name in ['temp-l2', 'temp-l1']:
                if (name,'mask') in self.weight_dict.keys():
                    loss = self.loss_dict['temp-l2:mask'](pred_mask, prev_pred_mask)
                    loss_values[(name,'mask')] = loss.item()
                    generator_loss += self.weight_dict[(name,'mask')] * loss
                if (name,'normal') in self.weight_dict.keys():
                    loss = self.loss_dict['temp-l2:normal'](
                        pred_normal*gt_mask_clamp, 
                        prev_pred_normal*gt_mask_clamp)
                    loss_values[(name,'normal')] = loss.item()
                    generator_loss += self.weight_dict[(name,'normal')] * loss
                if (name,'ao') in self.weight_dict.keys():
                    prev_pred_ao = prev_pred_warped[:,5:6,:,:]
                    loss = self.loss_dict['temp-l2:ao'](
                        pred_ao*gt_mask_clamp, 
                        prev_pred_ao*gt_mask_clamp)
                    loss_values[(name,'ao')] = loss.item()
                    generator_loss += self.weight_dict[(name,'ao')] * loss
                if (name,'depth') in self.weight_dict.keys():
                    prev_pred_depth = prev_pred_warped[:,4:5,:,:]
                    loss = self.loss_dict['temp-l2:depth'](
                        pred_depth*gt_mask_clamp, 
                        prev_pred_depth*gt_mask_clamp)
                    loss_values[(name,'depth')] = loss.item()
                    generator_loss += self.weight_dict[(name,'depth')] * loss
                if (name,'color') in self.weight_dict.keys():
                    prev_pred_color = self.shading(prev_pred_warped)
                    loss = self.loss_dict['temp-l2:color'](pred_color, prev_pred_color)
                    loss_values[(name,'color')] = loss.item()
                    generator_loss += self.weight_dict[(name,'color')] * loss

        return generator_loss, loss_values

    """deprecated"""
    def evaluate_discriminator(self, 
                               current_input, previous_input,
                               current_output, previous_output):
        """
        Discriminator takes the following inputs:
            - current input upsampled (B x 5 (mask+normal+depth) x H x W)
            - previous input warped upsampled (B x 5 x H x W)
            - current prediction with color (B x 8 (mask+normal+ao+color) x H x W)
            - previous prediction warped with color (B x 8 x H x W)
        All tensors are already padded
        Returns the score of the discriminator, averaged over the batch
        """
        assert current_input.shape[1] == 5
        assert previous_input.shape[1] == 5
        assert current_output.shape[1] == 8
        assert previous_output.shape[1] == 8
        B, _, H, W = current_input.shape

        input = torch.cat([current_input, previous_input, current_output, previous_output], dim=1)
        return self.adv_loss(self.discriminator(input))

    def train_discriminator(self, input, gt_high, 
                            previous_input, gt_prev_warped,
                            pred_high, pred_prev_warped,
                            use_checkpoint = False):
        """
        All inputs are in high resolution.
        input: B x 5 x H x W (mask+normal+depth)
        gt_high: B x 6 x H x W
        previous_input: B x 5 x H x W
        gt_prev_warped: B x 6 x H x W
        pred_high: B x 6 x H x W
        pred_prev_warped: B x 6 x H x W
        """

        assert self.has_discriminator

        def colorize_and_pad(tensor):
            assert tensor.shape[1]==6
            mask = tensor[:,0:1,:,:]
            normal = ScreenSpaceShading.normalize(tensor[:,1:4,:,:], dim=1)
            depth_ao = tensor[:,4:6,:,:]
            ao = tensor[:,5:6,:,:]
            color = self.shading(torch.cat([mask, normal, depth_ao], dim=1))
            tensor_with_color = torch.cat([mask, normal, ao, color], dim=1)
            return LossNetUnshaded.pad(tensor_with_color, self.padding)

        # assemble input
        input = LossNetUnshaded.pad(input, self.padding)
        gt_high = colorize_and_pad(gt_high)
        pred_high = colorize_and_pad(pred_high)
        previous_input = LossNetUnshaded.pad(previous_input, self.padding)
        gt_prev_warped = colorize_and_pad(gt_prev_warped)
        pred_prev_warped = colorize_and_pad(pred_prev_warped)

        # compute losses
        discr_loss = 0
        gt_score = 0
        pred_score = 0

        if ('adv','all') in self.weight_dict.keys(): # spatial-temporal
            gt_input = torch.cat([
                input, 
                previous_input, 
                gt_high, 
                gt_prev_warped], dim=1)
            pred_input = torch.cat([
                input, 
                previous_input, 
                pred_high, 
                pred_prev_warped], dim=1)
            discr_loss0, gt_score0, pred_score0 = self.adv_loss.train_discr(
                gt_input, pred_input, self.discriminator)
            discr_loss += self.weight_dict[('adv','all')] * discr_loss0
            gt_score += self.weight_dict[('adv','all')] * gt_score0
            pred_score += self.weight_dict[('adv','all')] * pred_score0

        if ('tgan','all') in self.weight_dict.keys(): # temporal
            gt_input = torch.cat([
                gt_high, 
                gt_prev_warped], dim=1)
            pred_input = torch.cat([
                pred_high, 
                pred_prev_warped], dim=1)
            discr_loss0, gt_score0, pred_score0 = self.temp_adv_loss.train_discr(
                gt_input, pred_input, self.temp_discriminator)
            discr_loss += self.weight_dict[('tgan','all')] * discr_loss0
            gt_score += self.weight_dict[('tgan','all')] * gt_score0
            pred_score += self.weight_dict[('tgan','all')] * pred_score0

        if ('sgan','all') in self.weight_dict.keys(): # spatial-temporal
            gt_input = torch.cat([
                input, 
                gt_high], dim=1)
            pred_input = torch.cat([
                input, 
                pred_high], dim=1)
            discr_loss0, gt_score0, pred_score0 = self.spatial_adv_loss.train_discr(
                gt_input, pred_input, self.spatial_discriminator)
            discr_loss += self.weight_dict[('sgan','all')] * discr_loss0
            gt_score += self.weight_dict[('sgan','all')] * gt_score0
            pred_score += self.weight_dict[('sgan','all')] * pred_score0

        return discr_loss, gt_score, pred_score
