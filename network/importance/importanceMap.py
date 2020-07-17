import torch
from utils import newton, bisection
import logging

class ImportanceMap(torch.nn.Module):
    """
    Abstract importance sampler.
    It defines a call operator that compute the importance map
    """

    def __init__(self, upsampleFactor):
        super().__init__()
        """
        Initializes the upsampling factor
        """
        self._upsampleFactor = upsampleFactor

    def _upsample(self, img, factor=None):
        """
        upsample the specified heatmap image.
        If factor is None, the upsampling factor from the constructor is used
        """
        if factor is None:
            factor = self._upsampleFactor
        img = torch.nn.functional.interpolate(
            img.unsqueeze(1), scale_factor=factor, mode='bilinear')[:,0,:,:]
        return img

    @staticmethod
    def basic_normalize(importanceMap, min=0, mean=1, pad=0):
        """
        Normalizes the importanceMap of shape Batch * Height * Width
        to have a specified minimal value and mean value.
        The minimal value is simply added to the importanceMap.

        The input has to be in range [0,\infty), a softplus

        pad: number of pixels cropped from the border before
         taking the mean
        """
        mean = max(min, mean)
        B, H, W = importanceMap.shape

        importanceMap = torch.clamp(importanceMap, min=0.0)
        m = torch.mean(
            importanceMap[:, pad:-pad-1, pad:-pad-1],
            dim=[1,2], keepdim=True)
        #print("normalize, mean:", list(m.detach().cpu().numpy()))
        m = torch.clamp(m, min=1e-7)
        output = min + importanceMap * ((mean-min)/m)

        stats = {'mean' : m.detach()}
        return output, stats

    @staticmethod
    def softplus_normalize(importanceMap, min=0, mean=1, pad=0):
        """
        Normalizes the importanceMap of shape Batch * Height * Width
        to have a specified minimal value and mean value.
        The minimal value is simply added to the importanceMap.

        The input has arbitrary range.
        It is first passed through a Softplus and then normalized.
        The result will be in range [0,\infty)

        pad: number of pixels cropped from the border before
         taking the mean
        """
        mean = max(min, mean)
        B, H, W = importanceMap.shape

        importanceMap = torch.nn.functional.softplus(importanceMap)
        m = torch.mean(
            importanceMap[:, pad:-pad-1, pad:-pad-1],
            dim=[1,2], keepdim=True)
        #print("normalize, mean:", list(m.detach().cpu().numpy()))
        m = torch.clamp(m, min=1e-7)
        output = min + importanceMap * ((mean-min)/m)

        stats = {'mean' : m.detach()}
        return output, stats

    @staticmethod
    def sigmoid_normalize(importanceMap, min=0, mean=1, pad=0):
        """
        Normalizes the importanceMap of shape Batch * Height * Width
        to have a specified minimal value and mean value.

        The input has arbitrary range.
        It is then transformed through the function
            f(x) = min + (1-min)*sigmoid(x+c)
        The normalization method optimizes the constant 'c'
        It is first passed through a Softmax and then normalized.
        The result will be in range [0,\infty)

        pad: number of pixels cropped from the border before
         taking the mean
        """
        mean = max(min, mean)
        B, H, W = importanceMap.shape

        def sigmoidNormalize(X, c, min_value=0):
            return torch.sigmoid(X + c)*(1-min_value) + min_value
        def getOptimizationFunction(X, target, min_value):
            def f(c):
                X2 = sigmoidNormalize(X, c, min_value)
                current = torch.mean(X2)
                return current-target
            return f

        if not hasattr(ImportanceMap.sigmoid_normalize, "lastC"):
            ImportanceMap.sigmoid_normalize.lastC = 0.0

        sor_omega = 0.5

        finalC = torch.zeros(B,
            device=importanceMap.device, dtype=importanceMap.dtype)
        iterations = torch.zeros(B,
            device='cpu', dtype=torch.int32)
        outputs = [None]*B
        for b in range(B):
            # non-differentiable optimization
            # like an oracle for the final 'c'
            initialC = torch.tensor(0.0, #ImportanceMap.sigmoid_normalize.lastC, 
                                    device=importanceMap.device, dtype=importanceMap.dtype)
            max_iteration = 40
            # newton or bisection
            c, iterations[b] = bisection(getOptimizationFunction(
                importanceMap[b].detach(), mean, min), initialC, max_iteration = max_iteration)
            if iterations[b] >= max_iteration:
                logging.warn("Newton optimization did not converge after %d iterations"%max_iteration)
            c = c.detach()
            ImportanceMap.sigmoid_normalize.lastC = c.item()
            finalC[b] = c
            # differentiable normalization
            outputs[b] = sigmoidNormalize(importanceMap[b], c, min)
            if torch.isnan(outputs[b]).any():
                logging.warn("NaN!!!")

        output = torch.stack(outputs, dim=0)

        stats = {'c' : finalC, 'iterations' : iterations}
        return output, stats
