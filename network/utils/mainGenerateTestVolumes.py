import torch
import numpy as np
import math
import os

def createInvSphere(size:int, file : str):
    """
    Creates a sphere dataset with a density of 0 in the very center and 1 at the corners
    """
    print("Create sphere of size",size,"and save to",file)
    data = np.zeros((size, size, size), dtype=np.float32)
    s = size / 2
    f = 2 / math.sqrt((s**2) * 3)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                data[x,y,z] = f * math.sqrt((x-s)**2 + (y-s)**2 + (z-s)**2) - 1
    data = torch.from_numpy(data)
    torch.ops.renderer.create_volume_from_tensor(data)
    torch.ops.renderer.save_volume_to_binary(file)

def createSphere(size:int, file : str):
    """
    Creates a sphere dataset with a density of 1 in the very center and 0 at the corners
    """
    print("Create sphere of size",size,"and save to",file)
    data = np.zeros((size, size, size), dtype=np.float32)
    s = size / 2
    f = 1 / math.sqrt((s**2) * 3)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                data[x,y,z] = 1 - (f * math.sqrt((x-s)**2 + (y-s)**2 + (z-s)**2))
    data = torch.from_numpy(data)
    torch.ops.renderer.create_volume_from_tensor(data)
    torch.ops.renderer.save_volume_to_binary(file)

def createCube(
    size:int, # spatial dimension (voxels)
    scale:float, # size of the cube iin [0,1]
    rot:np.array,  # rotation of the cube
    file):

    print("Create cube of size",size,"and save to",file)
    data = np.zeros((size, size, size), dtype=np.float32)

    for ix in range(size):
        for iy in range(size):
            for iz in range(size):
                x = (ix/size)*2-1
                y = (iy/size)*2-1
                z = (iz/size)*2-1
                #todo: apply rotation

                #compute distance to cube
                d = math.sqrt(
                    max(0, abs(x)-scale)**2 +
                    max(0, abs(y)-scale)**2 +
                    max(0, abs(z)-scale)**2)
                data[ix,iy,iz] = 1 - d

    data = torch.from_numpy(data)
    torch.ops.renderer.create_volume_from_tensor(data)
    torch.ops.renderer.save_volume_to_binary(file)

if __name__ == "__main__":
    OUTPUT_FOLDER = '../../isosurface-super-resolution-data/volumes/test/'
    torch.ops.load_library("./Renderer.dll")

    rot = np.array([0,0,0,1])
    for s in [64,128,256]:
        for d in [0.1, 0.2, 0.5, 0.8]:
            createCube(s, d, rot, os.path.join(OUTPUT_FOLDER, "cube%03d_%.1f.cvol"%(s,d)))

    for s in [64,128,256]:
        createInvSphere(s, os.path.join(OUTPUT_FOLDER, "invSphere%03d.cvol"%s))
        createSphere(s, os.path.join(OUTPUT_FOLDER, "sphere%03d.cvol"%s))

    pass
