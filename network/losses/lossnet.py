import torch
import torch.nn as nn
import torch.nn.functional as F

from .lossbuilder import LossBuilder

class LossNet(nn.Module):
    """
    Main Loss Module.

    device: cpu or cuda
    losses: list of loss terms with weighting as string
    opt: command line arguments (Namespace object) with:
     - further parameters depending on the losses

    """
    def __init__(self, device, input_channels, output_channels, high_res, padding, losses, opt):
        super(LossNet, self).__init__()
        self.padding = padding
        if hasattr(opt, "upsample"):
            self.upsample = opt.upsample
        else:
            self.upsample = 'bilinear'
        self.input_channels = input_channels
        self.output_channels = output_channels
        #List of tuples (name, weight or None)
        self.loss_list = [(s.split(':')[0],s.split(':')[1]) if ':' in s else (s,None) for s in losses.split(',')]
        
        # Build losses and weights
        builder = LossBuilder(device)
        self.loss_dict = {}
        self.weight_dict = {}

        self.loss_dict['mse'] = builder.mse() #always use mse for psnr
        self.weight_dict['mse'] = 0.0

        content_layers = []
        style_layers = []
        self.discriminator = None
        self.has_discriminator = False
        for name,weight in self.loss_list:
            if 'mse'==name or 'l2'==name or 'l2_loss'==name:
                self.weight_dict['mse'] = float(weight) if weight is not None else 1.0
            elif 'inverse_mse'==name:
                self.loss_dict['inverse_mse'] = builder.inverse_mse()
                self.weight_dict['inverse_mse'] = float(weight) if weight is not None else 1.0
            elif 'fft_mse'==name:
                self.loss_dict['fft_mse'] = builder.fft_mse()
                self.weight_dict['fft_mse'] = float(weight) if weight is not None else 1.0
            elif 'l1'==name or 'l1_loss'==name:
                self.loss_dict['l1'] = builder.l1_loss()
                self.weight_dict['l1'] = float(weight) if weight is not None else 1.0
            elif 'tl2'==name or 'temp-l2'==name:
                self.loss_dict['temp-l2'] = builder.temporal_l2()
                self.weight_dict['temp-l2'] = float(weight) if weight is not None else 1.0
            elif 'tl1'==name or 'temp-l1'==name:
                self.loss_dict['temp-l1'] = builder.temporal_l1()
                self.weight_dict['temp-l1'] = float(weight) if weight is not None else 1.0
            elif 'perceptual'==name:
                content_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in opt.perceptualLossLayers.split(',')]
                #content_layers = [('conv_4',1), ('conv_12',1)]
                self.weight_dict['perceptual'] = float(weight) if weight is not None else 1.0
            elif 'texture'==name:
                style_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in opt.textureLossLayers.split(',')]
                #style_layers = [('conv_1',1), ('conv_3',1), ('conv_5',1)]
                self.weight_dict['texture'] = float(weight) if weight is not None else 1.0
            elif 'adv'==name or 'gan'==name:
                self.discriminator, self.adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    input_channels + (output_channels+1),
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = False
                self.discriminator_clip_weights = False
                self.has_discriminator = True
            elif 'wgan'==name:
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + output_channels,
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = False
                self.discriminator_clip_weights = True
                self.has_discriminator = True
            elif 'wgan-gp'==name: # Wasserstein-GAN with gradient penalty
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + output_channels,
                    opt,
                    gradient_penalty = True)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = False
                self.discriminator_clip_weights = False
                self.has_discriminator = True
            elif 'tadv'==name or 'tgan'==name: #temporal adversary
                self.discriminator, self.adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    input_channels + 2*output_channels,
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = False
                self.has_discriminator = True
            elif 'twgan'==name: #temporal Wassertein GAN
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + 2*output_channels,
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = True
                self.has_discriminator = True
            elif 'twgan-gp'==name: #temporal Wassertein GAN with gradient penalty
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + 2*output_channels,
                    opt,
                    gradient_penalty = True)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = False
                self.has_discriminator = True
            elif 'ssim'==name:
                self.loss_dict['ssim'] = builder.ssim_loss(4)
                self.weight_dict['ssim'] = float(weight) if weight is not None else 1.0
            elif 'dssim'==name:
                self.loss_dict['dssim'] = builder.dssim_loss(4)
                self.weight_dict['dssim'] = float(weight) if weight is not None else 1.0
            elif 'lpips'==name:
                self.loss_dict['lpips'] = builder.lpips_loss(4, 0, 1)
                self.weight_dict['lpips'] = float(weight) if weight is not None else 1.0
            else:
                raise ValueError('unknown loss %s'%name)

        if len(content_layers)>0 or len(style_layers)>0:
            self.pt_loss, self.style_losses, self.content_losses = \
                    builder.get_style_and_content_loss(dict(content_layers), dict(style_layers))

        self.loss_dict = nn.ModuleDict(self.loss_dict)
        print('Loss weights:', self.weight_dict)
        print("Has discriminator:", self.has_discriminator)

    def print_summary(self, gt_shape, pred_shape, input_shape, prev_pred_warped_shape, num_batches, device):
        #Print networks for VGG + Discriminator
        import torchsummary
        if 'perceptual' in self.weight_dict.keys() or 'texture' in self.weight_dict.keys():
            print('VGG (Perceptual + Style loss)')
            torchsummary.summary(self.pt_loss, gt_shape, 2*num_batches, device=device.type)
        if self.discriminator is not None:
            print('Discriminator:')
            res = gt_shape[1]
            if self.discriminator_use_previous_image:
                input_images_shape = (gt_shape[0]+1+input_shape[0]+prev_pred_warped_shape[0], res, res)
            else:
                input_images_shape = (gt_shape[0]+1+input_shape[0], res, res)
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

    def forward(self, gt, pred, input, prev_input_warped, prev_pred_warped, use_checkpoints = False):
        """
        gt: ground truth high resolution image (B x C=output_channels x 4W x 4H)
        pred: predicted high resolution image (B x C=output_channels x 4W x 4H)
        input: low resolution input image (B x C=input_channels x W x H)
               Only used for the discriminator, can be None if only the other losses are used
        prev_pred_warped: predicted image from the previous frame warped by the flow
               Shape: B x C x 4W x 4H
               with C = output_channels + 1 (warped mask)
               Only used for temporal losses, can be None if only the other losses are used
        """

        B, Cout, Hhigh, Whigh = gt.shape
        assert Cout == self.output_channels
        assert gt.shape == pred.shape
        if input is not None:
            B2, Cin, H, W = input.shape
            assert B == B2
            assert Cin == self.input_channels
        elif 'adv' in self.weight_dict.keys():
            raise ValueError("No input specified, but that is required by GAN losses")
        _, Cout2, _, _ = prev_pred_warped.shape
        #assert Cout2 == Cout + 1

        generator_loss = 0.0
        loss_values = {}

        # zero border padding
        gt = LossNet.pad(gt, self.padding)
        pred = LossNet.pad(pred, self.padding)
        if prev_pred_warped is not None:
            prev_pred_warped = LossNet.pad(prev_pred_warped, self.padding)

        # normal, simple losses, uses gt+pred
        for name in ['mse','inverse_mse','fft_mse','l1','ssim', 'dssim', 'lpips']:
            if name in self.weight_dict.keys():
                loss = self.loss_dict[name](gt, pred)
                loss_values[name] = loss.item()
                generator_loss += self.weight_dict[name] * loss

        # special losses: perceptual+texture, uses gt+pred
        if 'perceptual' in self.weight_dict.keys() or 'texture' in self.weight_dict.keys():
            style_weight=self.weight_dict.get('texture', 0)
            content_weight=self.weight_dict.get('perceptual', 0)

            def evalPerceptual(gt, pred, 
                               _style_losses=self.style_losses, 
                               _content_losses=self.content_losses):
                style_score = torch.zeros((), dtype=gt.dtype, device=gt.device, requires_grad=True)
                content_score = torch.zeros((), dtype=gt.dtype, device=gt.device, requires_grad=True)

                input_images = torch.cat([gt[:,0:3,:,:], pred[:,0:3,:,:]], dim=0) # no alpha
                self.pt_loss(input_images)

                for sl in _style_losses:
                    style_score = style_score + sl.loss
                for cl in _content_losses:
                    content_score = content_score + cl.loss

                return style_score, content_score

            if False: #use_checkpoints:
                # TODO: why does checkpointing not work here?
                style_score, content_score = torch.utils.checkpoint.checkpoint(
                    evalPerceptual, gt, pred)
            else:
                style_score, content_score = evalPerceptual(gt, pred)

            generator_loss += style_weight * style_score + content_weight * content_score
            if 'perceptual' in self.weight_dict.keys():
                loss_values['perceptual'] = content_score.item()
            if 'texture' in self.weight_dict.keys():
                loss_values['texture'] = style_score.item()

        # special: discriminator, uses input+pred+prev_pred_warped
        if 'adv' in self.weight_dict.keys():
            input_high = F.interpolate(input, 
                                       size=(gt.shape[-2],gt.shape[-1]),
                                      mode=self.upsample)
            if self.discriminator_use_previous_image:
                input_images = torch.cat([input_high, pred, prev_pred_warped], dim=1)
            else:
                input_images = torch.cat([input_high, pred], dim=1)
            input_images = LossNet.pad(input_images, self.padding)
            if use_checkpoints:
                gen_loss = self.adv_loss(torch.utils.checkpoint.checkpoint(self.discriminator, input_images))
            else:
                gen_loss = self.adv_loss(self.discriminator(input_images))
            loss_values['discr_pred'] = gen_loss.item()
            generator_loss += self.weight_dict['adv'] * gen_loss

        # special: temporal l2 loss, uses input (for the mask) + pred + prev_warped
        for name in ['temp-l1', 'temp-l2']:
            if name in self.weight_dict.keys():
                #pred_with_mask = torch.cat([
                #    pred,
                #    F.interpolate(input[:,3:4,:,:], size=(gt.shape[-2],gt.shape[-1]), mode=self.upsample)
                #    ], dim=1)
                #prev_warped_with_mask = prev_pred_warped
                #loss = self.loss_dict[name](pred_with_mask, prev_warped_with_mask)
                loss = self.loss_dict[name](pred, prev_pred_warped)
                loss_values[name] = loss
                generator_loss += self.weight_dict[name] * loss

        return generator_loss, loss_values

    def train_discriminator(self, input,
                            gt_high, prev_input, gt_prev_warped,
                            pred_high, pred_prev_warped,
                            use_checkpoint = False):
        """
        Let Cin = input_channels (RGB + mask + optional normal + depth)
        Let Cout = output_channels + 1 (RGB + mask)
        Expected shapes:
         - input: low-resolution input, B x Cin x H x W
         - gt_high: ground truth high res image, B x Cout x 4H x 4W
         - gt_prev_warped: ground truth previous image warped, B x Cout x 4H x 4W
         - pred_high: predicted high res image, B x Cout x 4H x 4W
         - pred_prev_warped: predicted previous high res image, warped, B x Cout x 4H x 4W
        Note that the mask of the high-resolution image is not part of the generator,
         but interpolated later.
        """

        B, Cin, H, W = input.shape
        assert Cin == self.input_channels
        B2, Cout, Hhigh, Whigh = gt_high.shape
        assert B2 == B
        assert Cout == self.output_channels
        assert gt_prev_warped.shape == gt_high.shape
        assert pred_high.shape == gt_high.shape
        assert pred_prev_warped.shape == gt_high.shape

        assert 'adv' in self.weight_dict.keys()
        B, Cout, Hhigh, Whigh = gt_high.shape

        # assemble input
        input_high = F.interpolate(input, size=(Hhigh, Whigh), mode=self.upsample)
        if self.discriminator_use_previous_image:
            gt_input = torch.cat([input_high, gt_high, gt_prev_warped], dim=1)
            pred_input = torch.cat([input_high, pred_high, pred_prev_warped], dim=1)
        else:
            gt_input = torch.cat([input_high, gt_high], dim=1)
            pred_input = torch.cat([input_high, pred_high], dim=1)
        gt_input = LossNet.pad(gt_input, self.padding)
        pred_input = LossNet.pad(pred_input, self.padding)

        discr_loss, gt_score, pred_score = self.adv_loss.train_discr(
            gt_input, pred_input, self.discriminator, use_checkpoint)
        return discr_loss, gt_score, pred_score

    def get_discr_parameters(self):
        assert self.has_discriminator
        return self.discriminator.parameters()

    def discr_train(self):
        assert self.has_discriminator
        self.discriminator.train()
    def discr_eval(self):
        assert self.has_discriminator
        self.discriminator.eval()