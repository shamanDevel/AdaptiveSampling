from importance.importanceMap import ImportanceMap

import torch
import torch.nn.functional as F

class GradientImportanceMap(ImportanceMap):
    """
    An importance sampler that uses the sum of the gradient norms of various channels.
    You can select the channels and their weights
    """

    def __init__(self, upsamplingFactor:float, *args):
        """
        Creates a new gradient importance sampler.
        As arguments, pass the upsampling factor
        followed by a tuple of (channel, weight) for each channel to consider.
        """
        super().__init__(upsamplingFactor)
        for arg in args:
            assert isinstance(arg, tuple)
            assert len(arg)==2
            x, y = arg
            assert isinstance(x, int)
            assert isinstance(y, (int, float))
        self._channels = args

    def forward(self, input):
        # uses second-order differences in the center
        shape = input.shape
        # derivative along X
        inputX = F.pad(input, (0,0,1,1), mode='replicate')
        gX = (inputX[:,:,2:,:] - inputX[:,:,:-2,:]) / 2
        gX = torch.cat([
            gX[:,:,:1,:] * 2,
            gX[:,:,1:-1,:],
            gX[:,:,-1:,:] * 2
            ], dim=2)
        #gX[:,:,0,:] = gX[:,:,0,:]+gX[:,:,0,:]
        #gX[:,:,-1,:]*=2
        # derivate along Y
        inputY = F.pad(input, (1,1,0,0), mode='replicate')
        gY = (inputY[:,:,:,2:] - inputY[:,:,:,:-2]) / 2
        gY = torch.cat([
            gY[:,:,:,:1] * 2,
            gY[:,:,:,1:-1],
            gY[:,:,:,-1:] * 2
            ], dim=3)
        #gY[:,:,:,0]*=2
        #gY[:,:,:,:-1]*=2
        # add gradients together
        # TODO: L2 norm instead of L1 used here?
        g = torch.abs(gX) + torch.abs(gY)
        # sum channels
        grad = torch.zeros((shape[0], shape[2], shape[3]), 
                           dtype=input.dtype, device=input.device)
        for (channel, weight) in self._channels:
            grad += weight * g[:,channel,:,:]
        return self._upsample(grad).unsqueeze(1)

