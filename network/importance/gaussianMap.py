from importance.importanceMap import ImportanceMap

import torch
import numpy as np

class GaussianImportanceMap(ImportanceMap):
    """
    An importance sampler that just returns an uniform importance everywhere.
    """

    def __init__(self, upsamplingFactor:float, fx:float, fy:float, variance:float):
        """
        Generates a single focus density as a 2D gaussian.
        fx, fy: fraction of the map where the focus lies.
        variance: variance in pixels of the gaze region
        """
        super().__init__(upsamplingFactor)
        self._fx = fx
        self._fy = fy
        self._variance = variance

    def forward(self, input):
        shape = input.shape
        width = shape[3]
        height = shape[2]
        x, y = np.meshgrid(np.linspace(0, width-1, width), 
                           np.linspace(0, height-1, height))
        variance = self._variance * np.sqrt(width**2 + height**2)
        dx = (x-width*self._fx)**2 / (2*((variance)**2))
        dy = (y-height*self._fy)**2 / (2*(variance**2))
        density = np.exp(-dx - dy)
        t = torch.as_tensor(density, dtype=input.dtype, device=input.device)
        map = torch.stack([t]*shape[0])
        return self._upsample(map).unsqueeze(1)
