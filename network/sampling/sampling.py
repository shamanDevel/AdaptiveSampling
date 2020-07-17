import numpy as np

"""Basic sampling"""

def sampleRandomPoints(N, width, height):
    """creates completely random 2D points"""
    return np.stack(
        [np.random.rand(N) * width,
        np.random.rand(N) * height],
        axis=1
    )

# TODO: poisson disk, stratisfied, ...