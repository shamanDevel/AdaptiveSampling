"""
Algorithms, learnable and analytic ones, that compute an importance map from a given low-resolution image.

"""

from importance.importanceMap import ImportanceMap
from importance.uniformMap import UniformImportanceMap
from importance.gaussianMap import GaussianImportanceMap
from importance.gradientMap import GradientImportanceMap
from importance.networkMap import NetworkImportanceMap
from importance.luminanceMap import LuminanceImportanceMap
from importance.adaptiveSmoothing import adaptiveSmoothing
from importance.fractionalInpainting import fractionalInpaint
from importance.pdeInpainting import pdeInpaint

import torch
import torch.nn.functional as F
import dataset.datasetUtils

class PostProcess(torch.nn.Module):
    def __init__(self, minValue, meanValue, postUpscale, normalizePadding, normalizationMode):
        super().__init__()
        self._minValue = minValue
        self._meanValue = meanValue
        self._postUpscale = postUpscale
        self._normalizePadding = normalizePadding
        self._normalizationMode = normalizationMode

    def importanceCenter(self):
        """
        Returns the value of the importance map that leads to a uniform importance
        if no normalization is used.
        This is used as target in the importance-map regularization to 
        keep the normalization in bounds.
        """
        if self._normalizationMode == 'basic':
            return 0.5
        elif self._normalizationMode == 'softplus':
            return 0.5
        elif self._normalizationMode == 'sigmoid':
            return 0.0
        else:
            raise ValueError("unknown value for normalizationMode: "+self._normalizationMode)

    def forward(self, img):
        assert len(img.shape)==4
        assert img.shape[1]==1
        #img = img.unsqueeze(1)
        #img = torch.clamp(img[:,0,:,:], min=0) # not needed anymore for the output (sigmoid / softplus)
        img = img[:,0,:,:]
        img = F.pad(img, [-self._normalizePadding]*4, 'constant', 0)
        if self._normalizationMode == 'basic':
            img, stats = ImportanceMap.basic_normalize(
                img, self._minValue, self._meanValue, 0)
        elif self._normalizationMode == 'softplus':
            img, stats = ImportanceMap.softplus_normalize(
                img, self._minValue, self._meanValue, 0)
        elif self._normalizationMode == 'sigmoid':
            img, stats = ImportanceMap.sigmoid_normalize(
                img, self._minValue, self._meanValue, 0)
        else:
            raise ValueError("unknown value for normalizationMode: "+self._normalizationMode)
        img = F.pad(img, [self._normalizePadding]*4, 'constant', self._minValue)
        if self._postUpscale == 1:
            return img, stats
        img = torch.nn.functional.interpolate(
            img.unsqueeze(1), scale_factor=self._postUpscale, mode='bilinear', align_corners=False)
        return img[:,0,:,:], stats

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import imageio
    import torch
    import torch.nn.functional as F
    import scipy.misc
    import numpy as np
    torch.ops.load_library("./Renderer.dll")
    torch.manual_seed(42)

    #filename = "NormalBackpack.png"
    #filename = "NormalEjecta.png"
    filename = "screenshots/final-iso/iso-ejecta-lowres-normal-small.png"
    upsampling_factor = 4

    device = torch.device('cpu')
    deviceCuda = torch.device('cuda')

    print("Load image")
    img = imageio.imread(filename, pilmode='RGB') / 255.0
    img = 2.0*img - 1.0
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    img = img.to(device=device, dtype=torch.float32)
    img = F.interpolate(img, scale_factor=upsampling_factor, mode='bilinear')
    originalImg = img
    img = F.interpolate(img, scale_factor=1/upsampling_factor, mode='area')
    print(img.shape)

    minValue = 0.05
    meanValue = 0.2
    def post(img):
        preMin = torch.min(img).item(); preMax = torch.max(img).item();
        img, _ = PostProcess(minValue, meanValue, 1, 0, 'basic')(img)
        postMin = torch.min(img).item(); postMax = torch.max(img).item();
        print("Normalize: pre=[%f, %f], post=[%f, %f]"%(preMin, preMax, postMin, postMax))
        return img

    def save_importance(name, img):
        img = np.clip(img[0].detach().cpu().numpy(), 0, 1)
        img = np.stack([img]*3, axis=2)
        imageio.imwrite(name, img)

    print("Compute importance maps")
    uniform = post(UniformImportanceMap(upsampling_factor, 0.5)(img))
    gaussian = post(GaussianImportanceMap(upsampling_factor, 0.4, 0.6, 0.2)(img))
    gradRed = post(GradientImportanceMap(upsampling_factor, (0,1))(img))
    gradGray = post(GradientImportanceMap(upsampling_factor, (0,0.2), (1,0.7), (2,0.1))(img))
    save_importance("ejecta-gradient.png", gradGray)
    with torch.no_grad():
        network = post(NetworkImportanceMap(upsampling_factor, 3, output_layer='softplus')(img))
    luminance = post(LuminanceImportanceMap(
        upsampling_factor, [0,1,2], (0,1),
        "..\\..\\tests\\luminance-contrast\\siggraph2019-matlab", 2) \
                     (img))
    save_importance("ejecta-luminance3.png", luminance)
    

    print("Smooth input image: adaptive smoothing")
    def smoothOnHeat1(heatmap):
        distances = 1.0 / heatmap
        return torch.ops.renderer.adaptive_smoothing(
            originalImg.to(device=deviceCuda), 
            distances.to(device=deviceCuda).unsqueeze(1), 
            0.5).to(device=device)
    smooth1Gray = smoothOnHeat1(gradGray)
    smooth1Gaussian = smoothOnHeat1(gaussian)
    smooth1Luminance = smoothOnHeat1(luminance)

    print("Smooth input image: fractional inpainting")
    def smoothOnHeat2(heatmap):
        return torch.ops.renderer.fast_inpaint_fractional(
            heatmap.to(device=deviceCuda), originalImg.to(device=deviceCuda)).to(device=device)
    smooth2Gray = smoothOnHeat2(gradGray)
    smooth2Gaussian = smoothOnHeat2(gaussian)
    smooth2Luminance = smoothOnHeat2(luminance)

    print("compute errors:")
    def printError(smooth, name):
        l1 = F.l1_loss(smooth, originalImg)
        l2 = F.mse_loss(smooth, originalImg)
        print("  %s: l1=%7.5f, l2=%7.5f; values (min,max,avg): (%.3f, %.3f, %.3f)"%(
            name, l1, l2, 
            torch.min(smooth).item(), torch.max(smooth).item(), torch.mean(smooth).item()))
    print(" adaptive gaussian:")
    printError(smooth1Gray,     "Gray    ")
    printError(smooth1Gaussian, "Gaussian")
    printError(smooth1Luminance,  "Luminance ")
    print(" fractional inpainting:")
    printError(smooth2Gray,     "Gray    ")
    printError(smooth2Gaussian, "Gaussian")
    printError(smooth2Luminance,  "Luminance ")

    #print("For testing: save image and heatmap")
    #imageio.imwrite("TestSmoothingInput.png", originalImg[0].cpu().numpy().transpose(1,2,0))
    #imageio.imwrite("TestSmoothingHeatmap1.png", gradGray[0].cpu().numpy())
    #imageio.imwrite("TestSmoothingHeatmap2.png", gaussian[0].cpu().numpy())

    print("Plot")
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True)
    axes[0,0].imshow(originalImg[0].cpu().numpy().transpose(1,2,0)); axes[0,0].set_title("Input")
    axes[0,1].imshow(uniform[0].cpu().numpy()); axes[0,1].set_title("Uniform")
    axes[0,2].imshow(gaussian[0].cpu().numpy()); axes[0,2].set_title("Gaussian")
    axes[1,0].imshow(gradRed[0].cpu().numpy()); axes[1,0].set_title("Grad Red")
    axes[1,1].imshow(gradGray[0].cpu().numpy()); axes[1,1].set_title("Grad Gray")
    axes[1,2].imshow(luminance[0].detach().cpu().numpy()); axes[1,2].set_title("Luminance")
    axes[2,0].imshow(smooth1Gray[0].cpu().numpy().transpose(1,2,0)); axes[2,0].set_title("Smooth Gray")
    axes[2,1].imshow(smooth1Gaussian[0].cpu().numpy().transpose(1,2,0)); axes[2,1].set_title("Smooth Gaussian")
    axes[2,2].imshow(smooth1Luminance[0].cpu().numpy().transpose(1,2,0)); axes[2,2].set_title("Smooth Network")
    axes[3,0].imshow(smooth2Gray[0].cpu().numpy().transpose(1,2,0)); axes[3,0].set_title("Smooth Gray: Inpainting")
    axes[3,1].imshow(smooth2Gaussian[0].cpu().numpy().transpose(1,2,0)); axes[3,1].set_title("Smooth Gaussian: Inpainting")
    axes[3,2].imshow(smooth2Luminance[0].cpu().numpy().transpose(1,2,0)); axes[3,2].set_title("Smooth Luminance: Inpainting")
    fig.set_tight_layout(True)
    plt.show()


class NormalizedAndColorConvertedImportanceNetwork(torch.nn.Module):
    """
    wraps a network with channel normalization and color space conversion.
    The network takes a tuple as input, performs the conversion only on the first tensor.
    The outputs of the network are not transformed (importance map).
    """

    def __init__(self, network : torch.nn.Module, *,
                 normalize : dataset.datasetUtils.Normalization.Normalize = None,
                 colorSpace : str = None):
        """
        Applies normalization if 'normalize' is not None.
        Applies color space transformation afterwards if 'colorSpace' is not None.
        """
        super().__init__()
        from utils import colorConversion as cc
        self._network = network
        self._normalize = normalize
        self._colorToSpace = cc.DvrColorToSpace(colorSpace != None, colorSpace or 'rgb')

    def forward(self, *args):
        args = list(args)
        assert len(args)>0

        if self._normalize is not None:
            args[0] = self._normalize(args[0])
        args[0] = self._colorToSpace(args[0])

        args = tuple(args)
        results = self._network(*args)
        return results