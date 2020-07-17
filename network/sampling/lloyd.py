import numpy as np

import sampling

class WeightedLloyd(object):
    """
    Performs the weighted Lloyd's Algorithm
    """

    def init(self, density, N, init=None, epsilon=1e-5):
        """
        initializes the algorithm with the given density map
        and target number of samples
        """
        self._density = density
        self._N = N
        if init is None:
            width, height = self._density.shape
            self._P = sampling.sampleRandomPoints(N, width, height)
        else:
            self._P = init
        self._prevP = self._P
        self._epsilon = epsilon

    def points(self):
        return self._P
    def previous_points(self):
        return self._prevP

    def step(self):
        """
        Performs a step of the optimization algorithm.
        Returns true if a convergence criterion is fullfilled.
        """
        width, height = self._density.shape
        self._prevP = self._P
        cells = sampling.bounded_voronoi(self._P, [0, width, 0, height])
        centroids = np.array([cell.integrate(self._density)[1] for cell in cells])
        change = np.linalg.norm(centroids - self._P)
        self._P = centroids
        print("Change:", change)
        return change < self._epsilon