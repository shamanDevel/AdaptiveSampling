import numpy as np
import torch
import matplotlib.pyplot as plt
from models import UNet

def main():
    device = torch.device("cuda")

    inputPath = "D:/VolumeSuperResolution-InputData/sparse-rendering/cleveland60/sample00004_sparse.npy"
    input = torch.from_numpy(np.load(inputPath)).to(device=device, dtype=torch.float32)
    print("input:", input.shape)
    B, C, H, W = input.shape

    mask = torch.abs(input[:,0:1,:,:])
    print("Filled pixels:", end='')
    for b in range(B):
        print(" %d"%int(torch.sum(mask[b])), end='')
    print("  of %d pixels"%(H*W))

    print("Now test different u-net configuations and report the filled pixels for the first batch")

    wf = 2
    with torch.no_grad():
        for depth in range(1, 10):
            unet = UNet(in_channels=C, depth=depth, wf=wf, padding='partial', return_masks=True)
            unet.to(device)
            output, masks = unet(input, mask)
            print("Depth:", depth)
            print("  output shape:", output.shape)
            print("  Layer: shape + filled pixels")
            for i,m in enumerate(masks):
                print("   ",m.shape," -> %d of %d pixels"%(int(torch.sum(m[0])), m.shape[2]*m.shape[3]))

    print("Save mask")
    img = mask[0,0].cpu().numpy()
    plt.imsave("unet-test-mask.png", img, cmap='Greys')
    #img = np.stack((img, img, img), axis=0)
    #pilImage = Image.fromarray(np.uint8(np.clip((img*255).transpose((1, 0, 2)), 0, 255)))
    #pilImage.save("unet-test-mask.png")

if __name__ == "__main__":
    main()