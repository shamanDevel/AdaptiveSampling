import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import copy
import json
from importance import RenderSettings


def adaptiveStepsizeRendering(
    stepsizes : torch.Tensor, 
    render_settings : List[RenderSettings],
    fd_epsilon : float = 1e-5):
    """
    Wrapper around torch.ops.render_adaptive_stepsize
    that supports gradient propagation.
    
    It is called as
    <code>rendering_output = adaptiveStepsizeRendering(stepsizes, renderSettings)</code>
    where stepsizes is a tensor of shape B*1*H*W with only positive values
    and rendering_output is a tensor of shape B*C*H*W with C=10 for DVR.

    How to get from the importance map to stepsizes:
      normalization = importance.PostProcess(...)
      normalized_importance,_ = normalization(importance_map)
      stepsizes = 1.0 / normalized_importance

    The dataloader specifies the viewport (AdaptiveStepsizeRendering.VIEWPORT),
    only those parts are rendered.

    Output channels:
     - 0,1,2: red, green, blue
     - 3: alpha
     - 4,5,6: normal xyz
     - 7: depth
     - 8,9: flow xy
    """

    # these variables are captured in the function below
    epsilon = fd_epsilon
    settings = [s.clone() for s in render_settings]

    class AdaptiveStepsizeRendering(torch.autograd.Function):

        @staticmethod
        def forward(ctx, stepsizes):
            assert len(stepsizes.shape) == 4, "stepsizes must be of shape B*1*H*W, but is %s"%importance_map.shape
            B, C, H, W = stepsizes.shape
            assert C == 1, "stepsizes must have only one channel, but is %d"%C

            original_device = stepsizes.device
            target_device = torch.device("cuda")
            stepsizes_gpu = stepsizes.to(device=target_device)

            # render
            outputs = []
            for b in range(B):
                settings[b].send() # updates renderer
                valueScaling = settings[b].VALUE_SCALING

                # check viewport for compatibility
                sW, sH = tuple(settings[b].RESOLUTION)
                viewStartX, viewStartY, viewEndX, viewEndY = tuple(settings[b].VIEWPORT)
                assert H == viewEndY - viewStartY, \
                    "size of the stepsizes (%s) does not match size of the viewport (%s)"%(
                        stepsizes.shape, AdaptiveStepsizeRendering.VIEWPORT)
                assert W == viewEndX - viewStartX, \
                    "size of the stepsizes (%s) does not match size of the viewport (%s)"%(
                        stepsizes.shape, AdaptiveStepsizeRendering.VIEWPORT)

                # pad stepsizes for viewport
                stepsizes_padded = F.pad(
                    stepsizes_gpu[b,0], 
                    [viewStartX, sW-viewEndX, viewStartY, sH-viewEndY],
                    mode='constant', value=1.0)
                assert stepsizes_padded.shape[0] == sH, "wrong padding"
                assert stepsizes_padded.shape[1] == sW, "wrong padding"

                # render
                o = torch.ops.renderer.render_adaptive_stepsize(stepsizes_padded)

                # undo viewport padding
                o = o[:, viewStartY:viewEndY, viewStartX:viewEndX]
                assert o.shape[1] == stepsizes.shape[2], "wrong padding"
                assert o.shape[2] == stepsizes.shape[3], "wrong padding"

                outputs.append(o)
            output = torch.stack(outputs, dim=0).to(device=original_device)

            ctx.save_for_backward(stepsizes, output)
            return output

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            stepsizes, output = ctx.saved_tensors
            B, C, H, W = output.shape

            original_device = stepsizes.device
            target_device = torch.device("cuda")
            stepsizes_gpu = stepsizes.to(device = target_device)

            grad_stepsizes = []
            for b in range(B):
                settings[b].send() # updates renderer
                valueScaling = settings[b].VALUE_SCALING

                sW, sH = tuple(settings[b].RESOLUTION)
                viewStartX, viewStartY, viewEndX, viewEndY = tuple(settings[b].VIEWPORT)

                # Input: original stepsize s, output gradient c'.
                # Output: stesize gradient s'
                # Let c=f(s) be the rendering with stepsize s. Per pixel:
                # s' = c' * (f(s+epsilon)-f(s))/epsilon
                # where * is the dot-product

                print("Backward", b)
                print("original stepsize:", stepsizes[b,0])
                print("pertubed stepsize:", stepsizes[b,0]+epsilon)

                stepsizes_padded = F.pad(
                    stepsizes_gpu[b,0] + epsilon, 
                    [viewStartX, sW-viewEndX, viewStartY, sH-viewEndY],
                    mode='constant', value=1.0)
                oe = torch.ops.renderer.render_adaptive_stepsize(stepsizes_padded)
                oe = oe.to(device = original_device)
                oe = oe[:, viewStartY:viewEndY, viewStartX:viewEndX]
                o = output[b]

                print("original output:", o)
                print("pertubed output:", oe)
                print("gradient output:", grad_output[b])

                grad_stepsize = torch.sum(
                    grad_output[b] * ((oe-o)/epsilon),
                    0, keepdim=True)
                grad_stepsizes.append(grad_stepsize)

            grad_stepsizes = torch.stack(grad_stepsizes, dim=0)
            return grad_stepsizes

    # construct and apply the function, inserts the backward-node into the graph
    return AdaptiveStepsizeRendering.apply(stepsizes)


def __setupExample():
    torch.ops.load_library("./Renderer.dll")
    print("Renderer loaded")

    # Load volume
    torch.ops.renderer.load_volume_from_binary("../test-data/Engine.cvol")

    # Load settings
    s = RenderSettings()
    with open("../test-data/Engine-settings.json", "r") as f:
        o = json.load(f)
        s.from_dict(o)
    print("Settings loaded")

    return s

def __renderImage(settings : RenderSettings):
    import matplotlib.pyplot as plt

    SIZE = 256
    T = 0.2
    s = settings.clone()
    s.VIEWPORT = [256,256,256+SIZE,256+SIZE]
    s.timestep = T
    stepsizes = 0.95 + 0.1*torch.rand(1, 1, SIZE, SIZE, dtype=torch.float32)
    rendering = adaptiveStepsizeRendering(stepsizes, [s])

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(stepsizes[0,0].cpu().numpy())
    axes[0].set_title("Stepsizes")
    axes[1].imshow(rendering[0,0:3].permute(1,2,0).cpu().numpy())
    axes[1].set_title("Rendering")

    plt.show()

def __test_gradient(settings : RenderSettings):
    SIZE = 1
    T = 0.2
    s = settings.clone()
    s.VIEWPORT = [256,256,256+SIZE,256+SIZE]
    s.timestep = T
    stepsizes = 0.95 + 0.1*torch.rand(1, 1, SIZE, SIZE, dtype=torch.float32)
    def func(stepsize):
        return adaptiveStepsizeRendering(stepsize, [s], 1e-3)

    print("Perform numerical gradient checking")
    torch.set_printoptions(threshold=10000)
    stepsizes.requires_grad = True
    torch.autograd.gradcheck(func, stepsizes, eps=1e-3)

if __name__ == "__main__":
    settings = __setupExample()
    #__renderImage(settings)
    __test_gradient(settings)