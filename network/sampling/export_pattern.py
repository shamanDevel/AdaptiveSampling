import numpy as np
from scipy.spatial import Delaunay

def export_pattern(points, width, height, filename):
    """
    Exports the points to file.
    The output file first prints the size of the 
     generating density map "width height" in one line,
    followed by the number of points and number of 
     delaunay triangles in the next line.
    Then follows the coordinates of the points, one point per line,
    and the three indices per triangle, one triangle per line.

    Parameters:
     - points: N*2 array of the point coordinates
     - width, height: size of the generating density map
       (i.e. bounding box of the points)
     - filename: the filename where to write the data to
    """

    tri = Delaunay(points)
    num_points = points.shape[0]
    num_tris = tri.simplices.shape[0]

    with open(filename, "w") as f:
        f.write("%d %d\n"%(width, height))
        f.write("%d %d\n"%(num_points, num_tris))
        for i in range(num_points):
            f.write("%f %f\n"%(points[i,0], points[i,1]))
        for i in range(num_tris):
            f.write("%d %d %d\n"%(tri.simplices[i,0], tri.simplices[i,1], tri.simplices[i,2]))

