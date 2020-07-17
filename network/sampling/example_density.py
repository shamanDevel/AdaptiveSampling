import numpy as np
import imageio

"""
Utilities for generating example density distributions
"""

def generate_single_focus(width, height, fx, fy, variance):
    """
    Generates a single focus density as a 2D gaussian.
    width, height: dimension of the map
    fx, fy: fraction of the map where the focus lies.
    variance: variance in pixels of the gaze region
    """

    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    #dx = (x-width*fx)**2 / (2*((variance*(width/height)**2)**2))
    dx = (x-width*fx)**2 / (2*((variance)**2))
    dy = (y-height*fy)**2 / (2*(variance**2))
    density = np.exp(-dx - dy)
    return density

def density_from_image_gray(filename):
    """
    Loads an image, converts it to grayscale and uses that as density
    """
    img = imageio.imread(filename, as_gray=True)
    #img = np.swapaxes(img, 0, 1)
    return img

def density_from_image_grad(filename):
    """
    Loads an image, computes the gradient norm of the colors and uses that as density
    """
    img = imageio.imread(filename, pilmode='RGB') / 255.0
    #img = np.swapaxes(img, 0, 1)
    g1, g2 = np.gradient(img, axis=(0,1))
    g1 = np.abs(g1)
    g2 = np.abs(g2)
    g = np.sum(g1, axis=2)+np.sum(g2, axis=2)
    return g

