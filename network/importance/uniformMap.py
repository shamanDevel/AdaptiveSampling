from importance.importanceMap import ImportanceMap

import torch

class UniformImportanceMap(ImportanceMap):
    """
    An importance Map that just returns an uniform importance everywhere.
    """

    def __init__(self, upsamplingFactor:float, value=0.5):
        super().__init__(upsamplingFactor)
        self._value = value

    def forward(self, input):
        shape = input.shape
        map = self._value * torch.ones((shape[0], shape[2], shape[3]), 
                                  dtype=input.dtype, device=input.device)
        return self._upsample(map).unsqueeze(1)
