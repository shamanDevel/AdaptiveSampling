import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
plt.rcParams['figure.figsize'] = [15, 7.5]

from shutil import copyfile
copyfile("../../build/renderer/RelWithDebInfo/Renderer.dll", "Renderer.dll")

import imageio
inputMask = imageio.imread("inpainting-mask.png")[:,:,0]/255.0
print(inputMask.shape)
inputNormals = imageio.imread("inpainting-normals.png").transpose((2,0,1))/255.0
print(inputNormals.shape)
print("min:", np.min(inputNormals[0:3,:,:]), ", max:", np.max(inputNormals[0:3,:,:]))

def show_image(axis, image, title=None):
    image = np.clip(image, 0, 1)
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
    if len(image.shape)==3: #rgb(a)
        if image.shape[0]==3 or image.shape[0]==4:
            axis.imshow(image[0:3,:,:].transpose(1,2,0)) #rgb(a)
        elif image.shape[0]==1:
            axis.imshow(image[0], norm=normalize) #gray (unsqueezed)
    else:
        axis.imshow(image, norm=normalize)
    if title is not None:
        axis.set_title(title)

#fig, ax = plt.subplots(nrows=1, ncols=2)
#show_image(ax[0], inputMask, "mask")
#show_image(ax[1], inputNormals, "normals")
#plt.rcParams['figure.figsize'] = [15, 7.5]

# test non-power of two sizes
startX, endX = 20, 111
startY, endY = 13, 97
inputMask3 = inputNormals[3,startX:endX,startY:endY]
inputNormals3 = inputNormals[:,startX:endX,startY:endY]

# try CUDA implementation
import torch
torch.ops.load_library("./Renderer.dll")

device = torch.device("cuda")
inputMask3_pytorch = torch.from_numpy(inputMask3).to(device).unsqueeze(0)
inputNormals3_pytorch = torch.from_numpy(inputNormals3).to(device).unsqueeze(0)
print(inputNormals3_pytorch.shape)

outputNormals3_pytorch = torch.ops.renderer.fast_inpaint(inputMask3_pytorch, inputNormals3_pytorch)[0]
outputNormals3 = outputNormals3_pytorch.cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=3)
show_image(ax[0], inputMask3, "mask")
show_image(ax[1], inputNormals3, "input")
show_image(ax[2], outputNormals3, "output")

plt.show()