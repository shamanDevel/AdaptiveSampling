import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from PIL import Image

from halton import HaltonSampler
from plastic import PlasticSampler
from uniform import UniformSampler
from regular import RegularSampler

if __name__ == "__main__":
    if 1:
        output_file = "../paper-vis20/big-figures/sampling/sampling_%s_64.png"
        size = 64
        threshold = 0.2
        postscale = 4
        sampler = [
            (HaltonSampler(), "halton"),
            (PlasticSampler(2), "plastic"),
            (UniformSampler(), "random"),
            (RegularSampler(2), "regular")
            ]

        for s, n in sampler:
            data = s.fill_image((size, size))
            #data[data>threshold*size*size] = 0
            #data = data * (1/threshold)

            cmap = plt.cm.hot
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            image = cmap(norm(data))
            dtype = image.dtype
            image = np.array(
                Image.fromarray((image*255).astype(np.uint8)).resize((size*postscale, size*postscale), Image.NEAREST))
            image = image.astype(dtype) / 255.0
            plt.imsave(output_file%n, image)

    else:
        plt.rcParams.update({'font.size': 42, 'xtick.labelsize' : 2})
        fig = plt.figure()
        fig.colorbar(plt.cm.ScalarMappable(
            norm = cls.Normalize(0.0, 1.0),
            cmap = plt.cm.hot))
        plt.show()