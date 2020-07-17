import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt
from math import floor, ceil, acos

class VoronoiCell:
    def __init__(self, generator : np.array, corner : np.array):
        """
        Creates a new voronoi cell.
        generator: a 2d-vector specifying the generator point
        corner: an N*2 matrix with the corner vertices
        """
        assert generator.shape == (2,)
        assert len(corner.shape) == 2
        assert corner.shape[1] == 2

        self._generator = generator
        self._corner = corner

    def generator(self):
        return self._generator
    def corner(self):
        return self._corner

    def inside(self, point):
        """
        Checks if the point is inside this cell
        """
        import matplotlib.path as mpltPath
        path = mpltPath.Path(self._corner)
        return path.contains_point(point)

    def integrate(self, density):
        """
        Integrates the density over the area of this cell.
        Let (width,height) be the shape of the density array.
        The bounding box of the voronoi cells is assumed to be [0,width]x[0,height].

        Let A be the area of this cell, let phi:R^2->R+ be the density map.
        This method computes the integrated density
          $$ \int_A phi(x) dx $$
        and the centroid
          $$ \int_A x*phi(x) dx $$ .

        density: a 2D-array of shape (width,height) that defines the density field.

        return: a tuple (density, centroid)
        """

        # Note: the vertices in the cell are ordered clockwise

        width, height = density.shape
        # compute bounds of the cell
        min_x = width-1
        min_y = height-1
        max_x = 0.0
        max_y = 0.0
        for c in self._corner:
            min_x = min(min_x, floor(c[0]))
            min_y = min(min_y, floor(c[1]))
            max_x = max(max_x, ceil(c[0]))
            max_y = max(max_y, ceil(c[1]))
        min_x = max(0, min_x-1)
        min_y = max(0, min_y-1)
        max_x = min(width-1, max_x+1)
        max_y = min(height-1, max_y+1)
        # sample the polygon
        sampled_density = 0
        sampled_centroid = np.array([0.0,0.0])
        ## validate order
        #def normalize(x):
        #    return x / np.linalg.norm(x)
        #angle = np.dot(normalize(self._corner[1]-self._corner[0]), 
        #               normalize(self._corner[2]-self._corner[0]))
        #print("Angle:", acos(angle))
        # create mask
        X, Y = np.meshgrid(np.arange(min_x, max_x+1), np.arange(min_y, max_y+1), indexing='ij')
        mask = np.ones(X.shape)
        for i in range(len(self._corner)):
            x0 = self._corner[i]
            x1 = self._corner[(i+1)%len(self._corner)]
            mask *= ((X-x0[0])*(x1[1]-x0[1]) - (Y-x0[1])*(x1[0]-x0[0]) <= 1) * 1.0

        # why is this needed?
        if np.sum(mask) < 1:
            # invert order
            mask = np.ones(X.shape)
            for i in range(len(self._corner)):
                x1 = self._corner[i]
                x0 = self._corner[(i+1)%len(self._corner)]
                mask *= ((X-x0[0])*(x1[1]-x0[1]) - (Y-x0[1])*(x1[0]-x0[0]) <= 1) * 1.0

        # evaluate integral
        masked_density = mask * density[min_x:max_x+1, min_y:max_y+1]
        sampled_density = np.sum(masked_density)
        sampled_centroid = np.array([
            np.sum(X * masked_density),
            np.sum(Y * masked_density)
            ])

        ## test
        #if sampled_density == 0:
        #    mask = np.ones(X.shape)
        #    for i in range(len(self._corner)):
        #        x0 = self._corner[i]
        #        x1 = self._corner[(i+1)%len(self._corner)]
        #        mask *= ((X-x0[0])*(x1[1]-x0[1]) - (Y-x0[1])*(x1[0]-x0[0]) <= 1) * 1.0
        #
        #        # test
        #        mask_test = mask.copy()
        #        for c in self._corner:
        #            mask_test[
        #                max(0, min(mask.shape[0]-1, int(c[0]) - min_x)), 
        #                max(0, min(mask.shape[1]-1, int(c[1])-min_y))] = 2
        #        plt.figure()
        #        plt.title("%d of %d"%(i+1, len(self._corner)))
        #        plt.imshow(mask_test, extent=(min_x, max_x, min_y, max_y))
        #        plt.show()
        
        ##OLD:
        #sampled_density_old = 0
        #sampled_centroid_old = np.array([0.0,0.0])
        #for x in range(min_x, max_x+1):
        #    for y in range(min_y, max_y+1):
        #        p = np.array([x, y])
        #        if self.inside(p):
        #            d = density[x,y]
        #            sampled_density_old += d
        #            sampled_centroid_old += d * p
        #print("density: new=", sampled_density, ", old=", sampled_density_old)
        #print("centroid: new=", sampled_centroid[0], sampled_centroid[1],
        #     ", old=", sampled_centroid_old[0], sampled_centroid_old[1])

        sampled_centroid /= sampled_density + 1e-7
        return sampled_density, sampled_centroid


def bounded_voronoi(points : np.array, bounding_box):
    """
    Computes a bounded voronoi diagram.
    points: N*2 matrix of points
    bounding_box: [x_min, x_max, y_min, y_max]
    returns: list of VoronoiCell
    """
    # source: https://stackoverflow.com/a/33602171/1786598

    num_points = points.shape[0]
    # Mirror points
    points_center = points
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # create the cells
    return [VoronoiCell(points[i], vor.vertices[vor.regions[vor.point_region[i]]]) for i in range(num_points)]

def plotVoronoiCells(cells, ax):
    # initial points
    ax.plot([cell.generator()[0] for cell in cells], [cell.generator()[1] for cell in cells], 'b.')
    # vertices
    for cell in cells:
        ax.plot(cell.corner()[:,0], cell.corner()[:,1], 'go')
    # ridges
    for cell in cells:
        vertices = np.array(list(cell.corner()) + [cell.corner()[0]])
        ax.plot(vertices[:,0], vertices[:,1], 'k-')