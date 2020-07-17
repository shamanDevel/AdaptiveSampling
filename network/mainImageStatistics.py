import argparse
import imageio
import torch
import numpy as np

from utils.psnr import PSNR
from utils.ssim import MSSSIM, SSIM
from losses.lpips import PerceptualLoss

def loadImage(path):
    print("Load", path)
    img = imageio.imread(path)
    if len(img.shape)==2:
        # grayscale image
        img = np.stack([img]*3, axis=2)
    #print(img.shape)
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0 # as float in [0,1]
    if img.shape[0]==4:
        # remove alpha -> blend to white
        white = np.ones((3, img.shape[1], img.shape[2]), dtype=np.float32)
        alpha = img[3:4,:,:]
        img = alpha*img[0:3,:,:] + (1-alpha)*white
    # to PyTorch
    return torch.from_numpy(img).unsqueeze(0)

class Stats:
    def __init__(self, lpipsModel="net-lin", lpipsNet="alex"):
        self._psnr = PSNR()
        self._ssim = SSIM(val_range=1)
        self._msssim = MSSSIM()
        self._lpips = PerceptualLoss(
            lpipsModel, lpipsNet, use_gpu=True, gpu_ids=[0])
        self._device = torch.device("cuda")

    def run(self, img1, img2):
        img1 = img1.to(self._device)
        img2 = img2.to(self._device)
        with torch.no_grad():
            results = {
                "psnr" : self._psnr(img1, img2).item(),
                "ssim" : self._ssim(img1, img2).item(),
                "ms-ssim" : self._msssim(img1, img2).item(),
                "lpips" : self._lpips(img1*2-1, img2*2-1).item()
                }
            return results

def commandLine():
    """
    Computes image statistics between two RGB images.
    """

    parser = argparse.ArgumentParser(
        description="Image Statistics")
    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    parser.add_argument("--lpipsModel", type=str, default="net-lin")
    parser.add_argument("--lpipsNet", type=str, default="alex")
    opt = parser.parse_args()

    # Load images
    #print("Load", opt.img1)
    #print("Load", opt.img2)
    img1 = loadImage(opt.img1)
    img2 = loadImage(opt.img2)

    # compute statistics
    with torch.no_grad():
        results = Stats(lpipsModel=opt.lpipsModel, lpipsNet=opt.lpipsNet).run(img1, img2)

        print("PSNR:   ", results["psnr"])
        print("SSIM:   ", results["ssim"])
        print("MS-SSIM:", results["ms-ssim"])
        print("LPIPS:  ", results["lpips"])

def batchProcess():
    stats = Stats()

    if 0:
        folder = "screenshots/final-iso/"
        sets = ["ejecta", "ppmt1", "ppmt2", "skull"]
        classes = ["normal", "color"]
        gt = "gt"
        recons = ["interpolated", "rnet"]
        format = "iso-%s-%s-%s.png"
    elif 0:
        folder = "screenshots/final-dvr/"
        sets = ["ejecta3", "ppmt2", "thorax1"]
        gt = "gt"
        recons = ["interpolated", "rnet"]
        format = "dvr-%s-%s.png"
        classes = None
    elif 0:
        folder = "D:\\Promotion\\isosurface-data\\final-dvr2\\"
        sets = ["dvr10-ejecta4", "dvr15-ejecta4"] # dvr** with ** being the % of samples
        gt = "gt"
        recons = ["baseline", "rnet5", "rnet6pr", "rnet6v2"]
        format = "%s-%s.png"
        classes = None
    elif 0:
        folder = "screenshots/final-iso2/"
        sets = ["aneurism", "bug", "human", "jet"]
        classes = ["normal", "color"]
        gt = "gt"
        recons = ["interp", "recon"]
        format = "%s-%s-%s.png"
    elif 1:
        folder = "screenshots/final-dvr2/"
        sets = ["aneurism", "bug", "human", "jet"]
        gt = "gt"
        recons = ["interp", "recon"]
        format = "%s-%s.png"
        classes = None

    with open(folder+"stats.csv", "w") as f:
        f.write("Image1; Image2; PSNR; SSIM; MS-SSIM; LPIPS\n")
        for set in sets:
            if classes is None:
                gt_img_path = format%(set, gt)
                gt_img = loadImage(folder+gt_img_path)
                for recon in recons:
                    pred_img_path = format%(set, recon)
                    pred_img = loadImage(folder+pred_img_path)
                    results = stats.run(gt_img, pred_img)
                    f.write("%s; %s; %3.5f; %.5f; %.5f; %.5f\n"%(
                        gt_img_path, pred_img_path,
                        results['psnr'], results['ssim'], 
                        results['ms-ssim'], results['lpips']
                        ))
            else:
                for cls in classes:
                    gt_img_path = format%(set, gt, cls)
                    gt_img = loadImage(folder+gt_img_path)
                    for recon in recons:
                        pred_img_path = format%(set, recon, cls)
                        pred_img = loadImage(folder+pred_img_path)
                        results = stats.run(gt_img, pred_img)
                        f.write("%s; %s; %3.5f; %.5f; %.5f; %.5f\n"%(
                            gt_img_path, pred_img_path,
                            results['psnr'], results['ssim'], 
                            results['ms-ssim'], results['lpips']
                            ))

if __name__ == "__main__":
    #commandLine()
    batchProcess()
