import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lossbuilder import LossBuilder
from utils import ScreenSpaceShading

class LossNetSparse(nn.Module):
    """
    Main Loss Module for unshaded data.

    device: cpu or cuda
    losses: list of loss terms with weighting as string
       Format: <loss>:<target>:<weighting>
       with: loss in {l1, l2, tl2}
             target in {mask, normal, color, ao}
             weighting a positive number
    opt: command line arguments (Namespace object) with:
     - further parameters depending on the losses

    """
    def __init__(self, device, channels, losses, opt, has_flow):
        super().__init__()
        self.padding = opt.lossBorderPadding
        self.has_flow = has_flow
        if has_flow:
            assert channels == 8 # mask, normalX, normalY, normalZ, depth, ao, flowX, floatY
        else:
            assert channels == 6 # mask, normalX, normalY, normalZ, depth, ao
        #List of tuples (name, weight or None)
        self.loss_list = [s.split(':') for s in losses.split(',')]
        
        # Build losses and weights
        builder = LossBuilder(device)
        self.loss_dict = {}
        self.weight_dict = {}

        self.loss_dict['mse'] = builder.mse() #always use mse for psnr
        self.weight_dict[('mse','color')] = 0.0

        TARGET_INFO = {
            # target name : (#channels, expected min, expected max)
            "mask" : (1, -1, +1),
            "normal" : (3, -1, +1),
            "color" : (3, 0, 1),
            "ao" : (1, 0, 1),
            "depth" : (1, 0, 1)
            }

        content_layers = []
        style_layers = []
        self.has_discriminator = False
        self.has_style_or_content_loss = False
        self.has_temporal_l2_loss = False
        for entry in self.loss_list:
            if len(entry)<2:
                raise ValueError("illegal format for loss list: " + entry)
            name = entry[0]
            target = entry[1]
            weight = entry[2] if len(entry)>2 else None
            if target!='mask' and target!='normal' and target!='color' and target!='ao' and target!='depth' and target!='flow':
                raise ValueError("Unknown target: " + target)

            if 'mse'==name or 'l2'==name or 'l2_loss'==name:
                self.weight_dict[('mse',target)] = float(weight) if weight is not None else 1.0
            elif 'l1'==name or 'l1_loss'==name:
                self.loss_dict['l1'] = builder.l1_loss()
                self.weight_dict[('l1',target)] = float(weight) if weight is not None else 1.0
            elif 'tl2'==name or 'temp-l2'==name:
                self.loss_dict['temp-l2'] = builder.mse()
                self.weight_dict[('temp-l2',target)] = float(weight) if weight is not None else 1.0
                self.has_temporal_l2_loss = True
            elif 'tl1'==name or 'temp-l1'==name:
                self.loss_dict['temp-l1'] = builder.l1_loss()
                self.weight_dict[('temp-l1',target)] = float(weight) if weight is not None else 1.0
                self.has_temporal_l2_loss = True
            elif 'bounded'==name:
                if target!='mask':
                    raise ValueError("'bounded' loss can only be applied on the mask")
                self.weight_dict[("bounded", "mask")] = float(weight) if weight is not None else 1.0
            elif 'bce'==name:
                if target!='mask':
                    raise ValueError("'bce' loss can only be applied on the mask")
                self.weight_dict[("bce", "mask")] = float(weight) if weight is not None else 1.0
            elif 'ssim'==name:
                self.loss_dict['ssim'] = builder.ssim_loss(TARGET_INFO[target][0])
                self.weight_dict[('ssim',target)] = float(weight) if weight is not None else 1.0
            elif 'dssim'==name:
                self.loss_dict['dssim'] = builder.dssim_loss(TARGET_INFO[target][0])
                self.weight_dict[('dssim',target)] = float(weight) if weight is not None else 1.0
            elif 'lpips'==name:
                self.loss_dict['lpips'] = builder.lpips_loss(*TARGET_INFO[target])
                self.weight_dict[('lpips',target)] = float(weight) if weight is not None else 1.0
            else:
                raise ValueError('unknown loss %s'%name)

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
    def forward(self, gt, pred, prev_pred_warped, no_temporal_loss : bool, use_checkpoints = False):
        """
        gt: ground truth high resolution image (B x C x H x W)
        pred: predicted high resolution image (B x C x H x W)
        prev_pred_warped: predicted image from the previous frame warped by the flow
               Shape: B x C x H x W
               Only used for temporal losses, can be None if only the other losses are used
        use_checkpoints: True if checkpointing should be used.
               This does not apply here since we don't have GANs or perceptual losses
        """

        # TODO: loss that penalizes deviation from the input samples

        B, C, H, W = gt.shape
        if self.has_flow:
            assert C == 8
        else:
            assert C == 6
        assert gt.shape == pred.shape

        # zero border padding
        gt = LossNetSparse.pad(gt, self.padding)
        pred = LossNetSparse.pad(pred, self.padding)
        if prev_pred_warped is not None:
            prev_pred_warped = LossNetSparse.pad(prev_pred_warped, self.padding)

        # extract mask and normal
        gt_mask = gt[:,0:1,:,:]
        gt_mask_clamp = torch.clamp(gt_mask*0.5 + 0.5, 0, 1)
        gt_normal = ScreenSpaceShading.normalize(gt[:,1:4,:,:], dim=1)
        gt_depth = gt[:,4:5,:,:]
        gt_ao = gt[:,5:6,:,:]
        if self.has_flow:
            gt_flow = gt[:,6:8,:,:]
        pred_mask = pred[:,0:1,:,:]
        pred_mask_clamp = torch.clamp(pred_mask*0.5 + 0.5, 0, 1)
        pred_normal = ScreenSpaceShading.normalize(pred[:,1:4,:,:], dim=1)
        pred_depth = pred[:,4:5,:,:]
        pred_ao = pred[:,5:6,:,:]
        if self.has_flow:
            pred_flow = pred[:,6:8,:,:]
        if prev_pred_warped is not None and self.has_temporal_l2_loss:
            prev_pred_mask = prev_pred_warped[:,0:1,:,:]
            prev_pred_mask_clamp = torch.clamp(prev_pred_mask*0.5 + 0.5, 0, 1)
            prev_pred_normal = ScreenSpaceShading.normalize(prev_pred_warped[:,1:4,:,:], dim=1)
            prev_pred_depth = prev_pred_warped[:,4:5,:,:]
            prev_pred_ao = prev_pred_warped[:,5:6,:,:]
            if self.has_flow:
                prev_pred_flow = prev_pred_warped[:,6:8,:,:]

        generator_loss = 0.0
        loss_values = {}

        # normal, simple losses, uses gt+pred
        for name in ['mse','l1','ssim', 'dssim','lpips']:
            if (name,'mask') in self.weight_dict.keys():
                loss = self.loss_dict[name](gt_mask, pred_mask)
                loss_values[(name,'mask')] = loss.item()
                generator_loss += self.weight_dict[(name,'mask')] * loss
            if (name,'normal') in self.weight_dict.keys():
                loss = self.loss_dict[name](gt_normal*gt_mask_clamp, pred_normal*gt_mask_clamp)
                if torch.isnan(loss).item():
                    test = self.loss_dict[name](gt_normal*gt_mask_clamp, pred_normal*gt_mask_clamp)
                loss_values[(name,'normal')] = loss.item()
                generator_loss += self.weight_dict[(name,'normal')] * loss
            if (name,'ao') in self.weight_dict.keys():
                loss = self.loss_dict[name](gt_ao*gt_mask_clamp, pred_ao*gt_mask_clamp)
                loss_values[(name,'ao')] = loss.item()
                generator_loss += self.weight_dict[(name,'ao')] * loss
            if (name,'depth') in self.weight_dict.keys():
                loss = self.loss_dict[name](gt_depth*gt_mask_clamp, pred_depth*gt_mask_clamp)
                loss_values[(name,'depth')] = loss.item()
                generator_loss += self.weight_dict[(name,'depth')] * loss
            if (name,'flow') in self.weight_dict.keys():
                # note: flow is not restricted to inside regions
                loss = self.loss_dict[name](gt_flow, pred_flow)
                loss_values[(name,'flow')] = loss.item()
                generator_loss += self.weight_dict[(name,'flow')] * loss
            if (name,'color') in self.weight_dict.keys():
                gt_color = self.shading(gt)
                pred_color = self.shading(pred)
                loss = self.loss_dict[name](gt_color, pred_color)
                loss_values[(name,'color')] = loss.item()
                generator_loss += self.weight_dict[(name,'color')] * loss

        if ("bounded", "mask") in self.weight_dict.keys():
            # penalizes if the mask diverges too far away from [0,1]
            zero = torch.zeros(1,1,1,1, dtype=pred_mask.dtype, device=pred_mask.device)
            loss = torch.mean(torch.max(zero, pred_mask*pred_mask-2))
            loss_values[("bounded", "mask")] = loss.item()
            generator_loss += self.weight_dict[("bounded", "mask")] * loss

        if ("bce", "mask") in self.weight_dict.keys():
            # binary cross entry loss between the unclamped masks
            loss = F.binary_cross_entropy_with_logits(pred_mask*0.5+0.5, gt_mask*0.5+0.5)
            loss_values[("bce", "mask")] = loss.item()
            generator_loss += self.weight_dict[("bce", "mask")] * loss

        # temporal l2 loss, uses input (for the mask) + pred + prev_warped
        if self.has_temporal_l2_loss and not no_temporal_loss:
            assert prev_pred_warped is not None
            for name in ['temp-l2', 'temp-l1']:
                if (name,'mask') in self.weight_dict.keys():
                    loss = self.loss_dict[name](pred_mask, prev_pred_mask)
                    loss_values[(name,'mask')] = loss.item()
                    generator_loss += self.weight_dict[(name,'mask')] * loss
                if (name,'normal') in self.weight_dict.keys():
                    loss = self.loss_dict[name](
                        pred_normal*gt_mask_clamp, 
                        prev_pred_normal*gt_mask_clamp)
                    loss_values[(name,'normal')] = loss.item()
                    generator_loss += self.weight_dict[(name,'normal')] * loss
                if (name,'ao') in self.weight_dict.keys():
                    prev_pred_ao = prev_pred_warped[:,5:6,:,:]
                    loss = self.loss_dict[name](
                        pred_ao*gt_mask_clamp, 
                        prev_pred_ao*gt_mask_clamp)
                    loss_values[(name,'ao')] = loss.item()
                    generator_loss += self.weight_dict[(name,'ao')] * loss
                if (name,'depth') in self.weight_dict.keys():
                    prev_pred_depth = prev_pred_warped[:,4:5,:,:]
                    loss = self.loss_dict[name](
                        pred_depth*gt_mask_clamp, 
                        prev_pred_depth*gt_mask_clamp)
                    loss_values[(name,'depth')] = loss.item()
                    generator_loss += self.weight_dict[(name,'depth')] * loss
                if (name,'flow') in self.weight_dict.keys():
                    prev_pred_depth = prev_pred_warped[:,4:5,:,:]
                    loss = self.loss_dict[name](
                        pred_flow, #note: no restriction to inside areas
                        prev_pred_flow)
                    loss_values[(name,'flow')] = loss.item()
                    generator_loss += self.weight_dict[(name,'flow')] * loss
                if (name,'color') in self.weight_dict.keys():
                    prev_pred_color = self.shading(prev_pred_warped)
                    loss = self.loss_dict[name](pred_color, prev_pred_color)
                    loss_values[(name,'color')] = loss.item()
                    generator_loss += self.weight_dict[(name,'color')] * loss

        return generator_loss, loss_values

    
