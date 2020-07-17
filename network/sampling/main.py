import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import PIL

from voronoi import VoronoiCell, bounded_voronoi, plotVoronoiCells
from lloyd import WeightedLloyd
from lbg import WeightedLindeBuzoGray
from sampling import sampleRandomPoints
from example_density import generate_single_focus, density_from_image_gray, density_from_image_grad
from export_pattern import export_pattern

def simpleTest():
    width = 50
    height = 100
    #points = np.array([
    #    [0.2, 0.3],
    #    [0.7, 0.5],
    #    [0.6,0.2],
    #    [0.3, 0.9],
    #    [0.8, 0.1]
    #])
    points = sampleRandomPoints(100, width, height) #np.random.rand(20, 2)
    print("Points:"); print(points)
    
    cells = bounded_voronoi(points, [0,width,0,height])
    for i,cell in enumerate(cells):
        print("cell",i,"with point",cell.generator())
        print(cell.corner())

    # example density map
    density = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            density[x,y] = 1-math.sqrt((x/width-0.5)**2 + (y/height-0.5)**2)

    # compute cell integrals
    cell_densities = [cell.integrate(density) for cell in cells]

    # plotting
    plt.figure(figsize=(20,10))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(density, cmap=cm.hot)

    ax2 = plt.subplot(1, 2, 2)
    for cell, (d, _) in zip(cells, cell_densities):
        vertices = np.array(list(cell.corner()) + [cell.corner()[0]])
        ax2.fill(vertices[:,0], vertices[:,1], facecolor=cm.hot(d))
    ax2.plot([cell.generator()[0] for cell in cells], [cell.generator()[1] for cell in cells], 'b.')
    ax2.plot([d[1][0] for d in cell_densities], [d[1][1] for d in cell_densities], 'r.')
    for cell in cells:
        ax2.plot(cell.corner()[:,0], cell.corner()[:,1], 'go')
    for cell in cells:
        vertices = np.array(list(cell.corner()) + [cell.corner()[0]])
        ax2.plot(vertices[:,0], vertices[:,1], 'k-')
    #fig.show()
    plt.savefig("bounded_voronoi.png")

def simpleOptim():
    N = 200
    width = 100
    height = 100
    iterations = 10

    # create density
    density = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            density[x,y] = (1-math.sqrt(2)*math.sqrt((x/width-0.5)**2 + (y/height-0.5)**2))**2

    # create algorithms
    algs = [WeightedLloyd(), WeightedLindeBuzoGray()]
    algNames = ["Lloyd", "Linde-Buzo-Gray"]
    init_points = sampleRandomPoints(N, width, height)
    for alg in algs:
        alg.init(density, N, init_points)

    plt.figure(figsize=(5,5))
    ax1 = plt.subplot(1, 1, 1)
    ax1.imshow(density, cmap=cm.hot)
    plt.savefig("stipplingDensity.png")
    plt.close()

    # create animation
    def plot(cellsx, next_pointsx, algNames, filename):
        plt.figure(figsize=(10*len(algs),10))
        for i in range(len(algs)):
            ax2 = plt.subplot(1, len(algs), i+1)
            ax2.title.set_text(algNames[i])
            ax2.set_xlim(0, width)
            ax2.set_ylim(0, height)
            cells = cellsx[i]
            ax2.plot([cell.generator()[0] for cell in cells], [cell.generator()[1] for cell in cells], 'b.')
            ax2.plot(next_pointsx[i][:,0], next_pointsx[i][:,1], 'r.')
            for cell in cells:
                vertices = np.array(list(cell.corner()) + [cell.corner()[0]])
                ax2.plot(vertices[:,0], vertices[:,1], 'k-')
        plt.savefig(filename)
        plt.close()

    #plot(algs, algNames, "stippling%02d.png"%0)
    for i in range(iterations):
        cellsx = []
        next_pointsx = []
        for alg, algName in zip(algs, algNames):
            points = alg.points()
            cells = bounded_voronoi(points, [0,width,0,height])
            converged = alg.step()

            next_points = alg.points()
            cellsx.append(cells)
            next_pointsx.append(next_points)

            print(algName,"converged:",converged)
        plot(cellsx, next_pointsx, algNames, "stippling%02d.png"%i)

def optimDensity(density, N, name, scale=1):
    iterations = 10
    width, height = density.shape
    aspect = width / height
    print(width,",",height)

    plt.figure(figsize=(10/aspect,10))
    plt.imshow(density, cmap=cm.hot)
    plt.colorbar()
    plt.savefig("%s_density.png"%name)
    plt.close()

    alg =  WeightedLindeBuzoGray()
    img = PIL.Image.fromarray(density)
    density = np.array(img.resize((img.size[0]*scale, img.size[1]*scale), PIL.Image.BILINEAR))
    alg.init(density, N)

    def plot(points, filename):
        plt.figure(figsize=(20/aspect,20))
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(points[:,1]/scale, width-points[:,0]/scale-1, 'k.')
        ax1.set_ylim(0, width)
        ax1.set_xlim(0, height)
        plt.savefig(filename)
        plt.close()

    plot(alg.points(), "%s_points00.png"%name)
    for i in range(iterations):
        print("Iteration", (i+1))
        converged = alg.step()
        plot(alg.points(), "%s_points%02d.png"%(name, i+1))
        if converged:
            print("converged")
            break

    # save points
    print("Export")
    points2 = np.stack([
        alg.points()[:,1]/scale, width-alg.points()[:,0]/scale-1
        ], axis=1)
    export_pattern(points2, height, width, "%s_positions.txt"%name)
    #with open("%s_positions.txt"%name, "w") as f:
    #    px = alg.points()
    #    f.write("%d %d\n"%(height, width))
    #    for i in range(px.shape[0]):
    #        f.write('%f %f\n'%(px[i,1]/scale, width-px[i,0]/scale-1))

if __name__ == "__main__":
    #simpleTest()
    #simpleOptim()

    #optimDensity(generate_single_focus(500, 200, 0.5, 0.5, 50)+0.1, 500, "SingleGaze1")
    #optimDensity(generate_single_focus(1920*2, 1080*2, 0.5, 0.5, 200*2)+0.01, 100000, "SingleGaze2", scale=4)
    optimDensity(generate_single_focus(1920*2, 1080*2, 0.5, 0.5, 100*2)+0.01, 100000, "SingleGaze3", scale=4)

    #optimDensity(density_from_image_grad("NormalEjecta.png")+0.001, 5000, "Ejecta")
    #optimDensity(density_from_image_grad("NormalEjecta.png")+0.01, 50000, "Ejecta2", scale=4)