import numpy as np
import h5py
from halton import HaltonSampler
from plastic import PlasticSampler
from uniform import UniformSampler
from regular import RegularSampler

if __name__ == "__main__":
    output_file = "D:/VolumeSuperResolution-InputData/samplingPattern.hdf5"
    size = 2048

    sampler = [
        (HaltonSampler(), "halton"),
        (PlasticSampler(2), "plastic"),
        (UniformSampler(), "random"),
        (RegularSampler(2), "regular")
        ]

    with h5py.File(output_file, "w") as f:
        for s, n in sampler:
            print("Sampler:", n)
            dset = f.create_dataset(n, (size, size), dtype=np.float32)
            img = s.fill_image((size, size))
            img = (img.astype(np.float32) / (size * size)).astype(np.float32)
            print("min:", np.min(img), 
                  ", max:", np.max(img), 
                  ", mean:", np.mean(img), 
                  ", dtype:", img.dtype)
            dset[...] = img

    print("Done")
