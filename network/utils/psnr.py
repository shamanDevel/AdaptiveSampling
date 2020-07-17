
import torch
import torch.nn.functional as F
import numpy as np

class PSNR(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img1, img2, mask=None, epsilon=1e-7):
        """
        Computes the PSNR between the images img1 and img2, shape: BxCxHxW
        mask: optional mask in [0,1], all pixels with mask=0 are ignored, shape: Bx1xHxW
        """
        if mask is None:
            return 10 * torch.log10(1 / (epsilon + torch.mean((img1 - img2)**2, dim=[1,2,3])))
        else:
            img1 = mask * img1
            img2 = mask * img2
            B,C,H,W = mask.shape
            factor = (H*W) / torch.sum(mask, dim=[1,2,3])
            return 10 * factor * torch.log10(1 / (epsilon + torch.mean((img1 - img2)**2, dim=[1,2,3])))
