import torch
import numpy as np

class AdaptiveSmoothing(torch.autograd.Function):
    """
    Wrapper around torch.ops.renderer.adaptive_smoothing
    that supports gradient propagation.
    It is aliased by 'adaptiveSmooting', use it as
    <code>output = adaptiveSmoothing(input : Tensor, distances : Tensor, stddev : float)</code>
    Gradients can be computed with respect to 'input' and 'distances'.
    This requires the Renderer library to be loaded
    """

    @staticmethod
    def forward(ctx, input, distances, distanceToStandardDeviation:float):
        stddev_tensor = torch.from_numpy(np.array(distanceToStandardDeviation))
        ctx.save_for_backward(input, distances, stddev_tensor)
        return torch.ops.renderer.adaptive_smoothing(
            input, distances, distanceToStandardDeviation)

    @staticmethod
    def backward(ctx, grad_output):
        input, distances, stddev_tensor = ctx.saved_tensors
        distanceToStandardDeviation = stddev_tensor.item()
        grad_input = grad_distances = grad_stddev = None

        grad_input, grad_distances = torch.ops.renderer.adjoint_adaptive_smoothing(
            input, distances, distanceToStandardDeviation,
            grad_output,
            ctx.needs_input_grad[0], ctx.needs_input_grad[1])

        return grad_input, grad_distances, grad_stddev

adaptiveSmoothing = AdaptiveSmoothing.apply

if __name__ == "__main__":
    from gaussianMap import GaussianImportanceMap
    from importanceMap import ImportanceMap
    import imageio
    import torch.nn.functional as F
    torch.ops.load_library("./Renderer.dll")

    #filename = "NormalEjecta.png"
    #img = imageio.imread(filename, pilmode='RGB') / 255.0
    #img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    #img = img.to(device=torch.device('cuda'), dtype=torch.float64)
    #img = torch.nn.functional.interpolate(img, scale_factor=0.01, mode='area')
    img = torch.rand(1, 3, 8, 8, dtype=torch.float64, device=torch.device("cuda"))
    print(img.shape)
    original = img.clone().detach()

    minValue = 0.5 # in pixels
    meanValue = 3 # in pixels
    def post(img):
        preMin = torch.min(img).item(); preMax = torch.max(img).item();
        img = torch.clamp(img, min=0)
        img = ImportanceMap.normalize(img, minValue, meanValue)
        #img = torch.clamp(img, max=1)
        postMin = torch.min(img).item(); postMax = torch.max(img).item();
        print("Normalize: pre=[%f, %f], post=[%f, %f]"%(preMin, preMax, postMin, postMax))
        return img.unsqueeze(1)
    gaussian = post(GaussianImportanceMap(1, 0.4, 0.6, 0.2)(img)).to(dtype=torch.float64)

    distanceToStandardDeviation = 0.5
    def func(img, heat):
        return adaptiveSmoothing(img, heat, distanceToStandardDeviation)

    img.requires_grad = True
    gaussian.requires_grad = True

    print("Propagate gradients")
    prediction = func(img, gaussian)
    loss = F.mse_loss(prediction, img)
    print("Loss:", loss.item())
    #loss.retain_grad()
    loss.backward()
    assert torch.all(img == original)

    print("Perform numerical gradient checking")
    torch.set_printoptions(threshold=10000)
    torch.autograd.gradcheck(func, (img, gaussian), nondet_tol=1e-5)