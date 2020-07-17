import torch
import numpy as np

class FractionalInpainting(torch.autograd.Function):
    """
    Wrapper around torch.ops.renderer.fast_inpaint_fractional
    that supports gradient propagation.
    It is aliased by 'fractionalInpainting', use it as
    <code>output = fractionalInpainting(input : Tensor, sample_mask : Tensor)</code>
    Gradients can be computed with respect to 'input' and 'sample_mask'.
    This requires the Renderer library to be loaded
    """

    @staticmethod
    def forward(ctx, input, sample_mask):
        input, sample_mask = input.contiguous(), sample_mask.contiguous()
        ctx.save_for_backward(input, sample_mask)
        return torch.ops.renderer.fast_inpaint_fractional(
            sample_mask.cuda(), input.cuda()).to(device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        input, sample_mask = ctx.saved_tensors
        grad_input = grad_sample_mask = None

        grad_sample_mask, grad_input = torch.ops.renderer.adj_fast_inpaint_fractional(
            sample_mask.cuda(), input.cuda(), grad_output.cuda())

        return grad_input.to(device=input.device), grad_sample_mask.to(device=input.device)

fractionalInpaint = FractionalInpainting.apply

def __test_gradient():
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
    img = torch.rand(2, 2, 8, 8, dtype=torch.float64, device=torch.device("cuda"))
    print(img.shape)
    original = img.clone().detach()

    minValue = 0.5 # in pixels
    meanValue = 3 # in pixels
    def post(img):
        preMin = torch.min(img).item(); preMax = torch.max(img).item();
        img = torch.clamp(img, min=0)
        #img = ImportanceMap.normalize(img, minValue, meanValue)
        #img = torch.clamp(img, max=1)
        postMin = torch.min(img).item(); postMax = torch.max(img).item();
        print("Normalize: pre=[%f, %f], post=[%f, %f]"%(preMin, preMax, postMin, postMax))
        return img
    gaussian = post(GaussianImportanceMap(1, 0.4, 0.6, 0.2)(img)).to(dtype=torch.float64)

    def func(img, heat):
        return fractionalInpaint(img, heat)

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
    torch.autograd.gradcheck(func, (img, gaussian), nondet_tol=1e-7)

def __test_image():
    """
    Test how the fractional impainting works
    with fraction samples of various steepness.
    (Steepness of the sampling sigmoid function)
    """

    from importance.gradientMap import GradientImportanceMap
    from importance.importanceMap import ImportanceMap
    from sampling.uniform import UniformSampler
    import imageio
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    torch.ops.load_library("./Renderer.dll")

    # load image
    filename = "screenshots/dense1_png/rm_gt_normal.png"
    img = imageio.imread(filename, pilmode='RGB') / 255.0
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    img = img.to(device=torch.device('cuda'), dtype=torch.float32)
    img = torch.nn.functional.interpolate(img, scale_factor=0.1, mode='area')
    print("Image loaded:", img.shape)
    B, C, H, W = img.shape

    # compute importance
    importance_map = GradientImportanceMap(1, (0,1), (1,1), (2,1))(img)
    min_importance = 0.01
    mean_importance = 0.2
    importance_map = ImportanceMap.normalize(
        importance_map, min_importance, mean_importance, 0)
    importance_map = importance_map.unsqueeze(1)
    print("Importance map computed")

    # generate sampling pattern
    sample_pattern = UniformSampler().fill_image((1, 1, H, W))
    sample_pattern = torch.from_numpy(sample_pattern) \
        .to(device=torch.device('cuda'), dtype=torch.float32) \
        / (H*W)
    print("Sample pattern created")

    # visualize sampling + reconstruction
    fig, axes = plt.subplots(nrows=3)
    f1 = axes[0].imshow(img[0,0].cpu().numpy()) #default
    f2 = axes[1].imshow(img[0].permute(1,2,0).cpu().numpy())
    f3 = axes[2].imshow(img[0].permute(1,2,0).cpu().numpy())
    axes[0].set_title("Pattern")
    axes[1].set_title("Sampling")
    axes[2].set_title("Interpolation")

    initial_steepness = 10
    max_steepness = 20
    # slider
    axslider = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axslider, 'Steepness', 0, max_steepness, valinit=initial_steepness)

    def update(val):
        # compute images
        print("update", val)
        if val < max_steepness:
            steepness = val**2
            sample_mask = torch.sigmoid(steepness * (importance_map - sample_pattern))
            print("fractional sampling with steepness", steepness)
            scaled_image = sample_mask * img
            impainted_image = torch.ops.renderer.fast_inpaint_fractional(
                sample_mask[0], img)
        else:
            sample_mask = (importance_map >= sample_pattern).to(dtype=importance_map.dtype)
            print("hard sampling")
            scaled_image = sample_mask * img
            impainted_image = torch.ops.renderer.fast_inpaint(
                sample_mask[0], img)
        # send to matplotlib
        f1.set_data(sample_mask[0,0].cpu().numpy())
        f2.set_data(scaled_image[0].permute(1,2,0).cpu().numpy())
        f3.set_data(impainted_image[0].permute(1,2,0).cpu().numpy())
        # redraw canvas while idle
        fig.canvas.draw_idle()

    samp.on_changed(update)
    update(initial_steepness)
    plt.title("Sampling Steepness test")
    plt.show()

if __name__ == "__main__":
    __test_gradient()
    #__test_image()
