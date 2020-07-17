import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTools:

    @staticmethod
    def flatten_high(image_high, upscale_factor):
        """
        Reshapes the high resolution input image with shape
          B x C x H*upscale_factor x W*upscale_factor
        to a low resolution output image with shape
          B x (C * upscale_factor * upscale_factor) x H x W .
        
        This operation is the inverse of PixelShuffle
        """
        #source: https://github.com/pytorch/pytorch/issues/2456
        b, c, h, w = image_high.shape
        r = upscale_factor
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        fm_view = image_high.contiguous().view(b, c, out_h, r, out_w, r)
        fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
        return fm_prime

    _offset_cache = dict()
    @staticmethod
    def _grid_offsets(H, W, dtype, device):
        """
        Returns the grid offsets HxWx2 within [-1,1]
        """
        if (H,W) in VideoTools._offset_cache:
            return VideoTools._offset_cache[(H,W)]
        else:
            print("Create grid offsets for warping: W=%d, H=%d"%(W, H))
            grid_offsetsH = torch.linspace(-1, +1, H, dtype=dtype, device=device)
            grid_offsetsW = torch.linspace(-1, +1, W, dtype=dtype, device=device)
            grid_offsetsH = torch.unsqueeze(grid_offsetsH, 1)
            grid_offsetsW = torch.unsqueeze(grid_offsetsW, 0)
            grid_offsets = torch.stack(
                torch.broadcast_tensors(grid_offsetsW, grid_offsetsH), 
                dim=2)
            grid_offsets = torch.unsqueeze(grid_offsets, 0) # batch dimension
            grid_offsets = grid_offsets.detach()
            VideoTools._offset_cache[(H,W)] = grid_offsets
            return grid_offsets

    @staticmethod
    #@profile
    def warp_upscale(image_high, flow_low, upscale_factor, special_mask=False):
        """
        Warps the high resolution input image with shape
          B x C x H*upscale_factor x W*upscale_factor
        with the upscaled low resolution flow in screen space with shape
          B x 2 x H x W.
        Output is the high resolution warped image

        If special_mask==True, the first channel is treated as being the mask in range [-1,+1].
        This channel is padded with -1, whereas all other channels with zero.
        """
        B, C, H, W = flow_low.shape
        assert C==2

        flow_x, flow_y = torch.chunk(flow_low, 2, dim=1)
        flow_x = flow_x * -2.0
        flow_y = flow_y * -2.0
        flow_low2 = torch.cat((flow_x, flow_y), dim=1)

        flow_high = F.interpolate(flow_low2, scale_factor = upscale_factor, mode='bilinear')
        flow_high = flow_high.permute(0, 2, 3, 1) # move channels to last position
        _, Hhigh, Whigh, _ = flow_high.shape

        grid_offsets = VideoTools._grid_offsets(Hhigh, Whigh, flow_high.dtype, flow_high.device)
        grid = grid_offsets + flow_high

        if special_mask:
            image_high = torch.cat([
                image_high[:,0:1,:,:]*0.5+0.5,
                image_high[:,1:,:,:]], dim=1)
        warped_high = F.grid_sample(image_high, grid)
        if special_mask:
            warped_high = torch.cat([
                warped_high[:,0:1,:,:]*2-1,
                warped_high[:,1:,:,]], dim=1)

        return warped_high

