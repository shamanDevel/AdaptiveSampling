import torch
import torch.nn.functional as F

class HistogramEqualization:
    """
    Performs histogram equalization.
    Inputs are tensor of shape (Batch, Pixels),
    hence the spatial dimension are flattened.
    If you want to equalize over channels, flatten it into the pixels;
    if you want the equalization to be independent per channel,
    flatten it into the batch.

    Source: https://en.wikipedia.org/wiki/Histogram_equalization
    """

    NUM_BINS = 256
    SMOOTHING_STEPS = 5

    @staticmethod
    def equalize(input):
        """
        Input: tensor of shape (Batch, Pixels)
        Output: tuple of
         - equalized output tensor of shape (Batch, Pixels)
         - an object which can be called to un-normalize the tensor.

        The tensor values are expected to be in range [0,1]

        Example:
          input_tensor = ....
          output_tensor, denormalization = HistogramEqualization.equalize(input_tensor)
          assert input_tensor == denormalization(output_tensor)
        """

        #assert torch.min(input) >= 0, "input tensor has to be in range [0,1]"
        #assert torch.max(input) <= 1, "input tensor has to be in range [0,1]"

        B, N = input.shape
        output = torch.empty_like(input)
        for b in range(B):
            # compute histogram
            histo = torch.histc(input[b], bins=HistogramEqualization.NUM_BINS, min=0, max=1)
            # smooth histogram
            kernel = torch.tensor(
                [[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]],
                dtype=input.dtype, device=input.device)
            histo = histo.unsqueeze(0).unsqueeze(0)
            histo = F.pad(
                histo, 
                [3*HistogramEqualization.SMOOTHING_STEPS, 3*HistogramEqualization.SMOOTHING_STEPS],
                'reflect')
            for i in range(HistogramEqualization.SMOOTHING_STEPS):
                histo = F.conv1d(histo, kernel)
            histo = histo[0,0,:]
            # cummulative histogram
            cum_histo = torch.cumsum(histo, 0, dtype=input.dtype)
            N = cum_histo[-1]
            # first non-zero value of the histogram
            cdf_min_pos = torch.argmax(
                (histo > 0) * \
                torch.linspace(1.0, 0.5, steps=HistogramEqualization.NUM_BINS, dtype=input.dtype, device=input.device))
            cdf_min = histo[cdf_min_pos].to(dtype=input.dtype)
            # compute new values for each bin
            heights = (cum_histo - cdf_min) / (N - cdf_min)
            # use these heights as basis for linear interpolation
            torch.ops.renderer.regular_interp1d(
                heights.unsqueeze(0), input[b:b+1], output[b:b+1])

        # TODO: denormalization
        return output, None

    

if __name__ == "__main__":
    import sys
    import h5py
    import numpy as np
    from skimage import color
    import matplotlib.pyplot as plt
    torch.ops.load_library("./Renderer.dll")
    np.set_printoptions(threshold=sys.maxsize)

    # test the max Luminosity
    luminosity_channel = 0 # for HSV
    white_rgb = np.ones((1, 1, 3), dtype=np.float32)
    white_lab = color.rgb2lab(white_rgb)
    max_luminosity = white_lab[0,0,luminosity_channel]
    print("Max luminosity:", max_luminosity)

    # Load test image
    FILE = "D:/VolumeSuperResolution-InputData/gt-dvr-ejecta2.hdf5"
    ENTRY = 86
    TIMESTEP = 5
    with h5py.File(FILE, 'r') as f:
        image = f['gt'][ENTRY, TIMESTEP, 0:3, :, :]
    image = image.transpose((1,2,0))
    H, W, _ = image.shape

    # convert to Lab
    image_lab = color.rgb2lab(image)

    # equalize luminosity
    luminosity = image_lab[:, :, luminosity_channel]/max_luminosity
    flat_luminosity = luminosity.reshape((1, H*W))
    flat_luminosity = torch.from_numpy(flat_luminosity)
    flat_luminosity_eq, denormalization = HistogramEqualization.equalize(flat_luminosity)
    flat_luminosity_eq = flat_luminosity_eq.numpy()
    luminosity_eq = (flat_luminosity_eq.reshape((H, W))) * max_luminosity

    # plug into image again
    image_lab_eq = np.copy(image_lab)
    image_lab_eq[:, :, luminosity_channel] = luminosity_eq
    image_eq = color.lab2rgb(image_lab_eq)
    #image_eq = np.clip(image_eq, 0.0, 1.0)

    # alternate equalization that simply normalizes luminosity
    luminosity_eq2 = (luminosity / np.max(luminosity)) * max_luminosity
    image_lab_eq2 = np.copy(image_lab)
    image_lab_eq2[:, :, luminosity_channel] = luminosity_eq2
    image_eq2 = color.lab2rgb(image_lab_eq2)

    # visualize
    fig, axes = plt.subplots(ncols=3)
    axes[0].imshow(image); axes[0].set_title("Input")
    axes[1].imshow(image_eq); axes[1].set_title("Equalized Histogram")
    axes[2].imshow(image_eq2); axes[2].set_title("Normalized Luminosity")
    plt.show()