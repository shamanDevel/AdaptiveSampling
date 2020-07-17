import torch
import torch.nn.functional as F
import numpy as np

def initialImage(current_input, channels, mode, aoInverted=False, upscaling=4):
    """
    For the first frame of the sequence,
    computes the high-res image that is 
    used as the previous image to the current network.
        - current_input: low resolution input, also specifies the dimension
        This is a tensor of shape B, Cin, Hlow, Wlow
        - channels: the expected number output channels
        - mode: the input image type, one of:
        - "zero" : all entries are zero
        - "unshaded" : for unshaded, mask=-1, normal=[0,0,1], depth=0.5, ao=1
        - "input": upsample current input, fill remaining channels
        - upscaling: the upscaling factor, default = 4
    """
        
    B, Cin, H, W = current_input.shape
    Hhigh = H * upscaling
    Whigh = W * upscaling

    if mode == "zero":
        return torch.zeros(B, channels, Hhigh, Whigh, 
                            dtype=current_input.dtype,
                            device=current_input.device)
    elif mode == "unshaded":
        if channels == 5:
            defaults = np.array([-1, 0, 0, 1, 0.5])
        elif channels == 6:
            defaults = np.array([-1, 0, 0, 1, 0.5, 0 if aoInverted else 1])
        elif channels == 8:
            defaults = np.array([-1, 0, 0, 1, 0.5, 0 if aoInverted else 1, 0, 0])
        else:
            raise ValueError("for mode='unshaded', channels is expected to be 5 or 6")
        defaults = torch.from_numpy(defaults).to(
            device=current_input.device, 
            dtype=current_input.dtype)
        defaults = defaults.view(1, channels, 1, 1)
        return defaults.expand(B, channels, Hhigh, Whigh)
    elif mode == "input":
        input_high = F.interpolate(current_input, scale_factor=upscaling, mode='bilinear')
        if channels == Cin:
            return input_high
        elif channels < Cin:
            return input_high[:,0:channels,:,:]
        else:
            return torch.cat([
                input_high,
                torch.ones(B, channels-Cin, Hhigh, Whigh,
                           dtype=current_input.dtype,
                           device=current_input.device)
                ], dim=1)
    else:
        raise ValueError("unknown input mode: " + mode)
