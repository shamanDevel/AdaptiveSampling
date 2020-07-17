import torch.nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm

def apply_sn(module):
    """
    Applies spectral normalization to all convolutional and linear layers in the given module
    """
    if isinstance(module, nn.Conv2d):
        SpectralNorm.apply(module, 'weight', 1, 0, 1e-12)
    elif isinstance(module, nn.Linear):
        SpectralNorm.apply(module, 'weight', 1, 0, 1e-12)
    elif isinstance(module, nn.Module):
        for c in module.children():
            apply_sn(c)