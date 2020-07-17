import numpy as np

import sampling

class WeightedLindeBuzoGray(object):
    """
    Performs the weighted Linde-Buzo-Gray (LBG) Algorithm
    """

    def init(self, density, N, init=None, epsilon=1):
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

        # compute bounds
        V = np.sum(density)
        T = V / N
        self._Tlow  = max(T/2, T - epsilon*V/N)
        self._Thigh = T + epsilon*V/N
        print("Weighted Linde-Buzo-Gray config:")
        print(" V:", V, ", N:", N, ", T:", T)
        print(" Tlow:", self._Tlow, ", Thigh:", self._Thigh)

    def points(self):
        return self._P
    def previous_points(self):
        return self._prevP

    def _twoPointsFurthestApartInCell(self, cell):
        dist = 0
        p1 = None
        p2 = None
        n = cell.corner().shape[0]
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(cell.corner()[i]-cell.corner()[j])
                if d > dist:
                    dist = d
                    p1 = cell.corner()[i]
                    p2 = cell.corner()[j]
        return p1, p2

    def step(self):
        """
        Performs a step of the optimization algorithm.
        Returns true if a convergence criterion is fullfilled.
        """
        width, height = self._density.shape
        self._prevP = self._P
        cells = sampling.bounded_voronoi(self._P, [0, width, 0, height])
        centroids = np.array([cell.integrate(self._density)[1] for cell in cells])
        points = []
        split_or_merge = False
        for cell in cells:
            d, c = cell.integrate(self._density)
            if d < self._Tlow:
                # delete point
                split_or_merge = True
            elif d > self._Thigh:
                # split cell
                p1, p2 = self._twoPointsFurthestApartInCell(cell)
                displacement = 0.25 * (p1 - p2)
                points.append(c - displacement)
                points.append(c + displacement)
                split_or_merge = True
            else:
                # move to centroid
                points.append(c)
        self._P = np.array(points)
        print("Num points:", len(points))
        return not split_or_merge