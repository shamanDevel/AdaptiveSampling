import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if __name__ == '__main__':
    device = torch.device("cuda")
    
    # rgb to hsv
    class RgbToHsv(nn.Module):
        def forward(self, image):
            r: torch.Tensor = image[..., 0, :, :]
            g: torch.Tensor = image[..., 1, :, :]
            b: torch.Tensor = image[..., 2, :, :]

            maxc: torch.Tensor = image.max(-3)[0]
            minc: torch.Tensor = image.min(-3)[0]

            v: torch.Tensor = maxc  # brightness

            deltac: torch.Tensor = maxc - minc
            s: torch.Tensor = deltac / v
            #s[torch.isnan(s)].fill_(0.0)
            s = torch.where(v==0, torch.zeros_like(s), s)

            # avoid division by zero
            deltac = torch.where(
                deltac == 0, torch.ones_like(deltac), deltac)

            rc: torch.Tensor = (maxc - r) / deltac
            gc: torch.Tensor = (maxc - g) / deltac
            bc: torch.Tensor = (maxc - b) / deltac

            maxg: torch.Tensor = g == maxc
            maxr: torch.Tensor = r == maxc

            h: torch.Tensor = 4.0 + gc - rc
            h[maxg] = 2.0 + rc[maxg] - bc[maxg]
            h[maxr] = bc[maxr] - gc[maxr]
            h[minc == maxc].fill_(0.0)

            h = (h / 6.0) % 1.0

            h = 2 * math.pi * h
            return torch.stack([h, s, v], dim=-3)
    rgb2hsv = RgbToHsv()
    rgb2hsv.to(device)
    rgb2hsv_scripted = torch.jit.script(rgb2hsv)
    test_out = rgb2hsv_scripted(torch.rand(2, 3, 10, 12, dtype=torch.float32, device=device))
    assert test_out.shape == (2, 3, 10, 12)
    assert test_out.dtype == torch.float32
    torch.jit.save(rgb2hsv_scripted, "rgb2hsv.pt")
    

    # hsv to rgb
    class HsvToRgb(nn.Module):
        def forward(self, image):
            h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
            s: torch.Tensor = image[..., 1, :, :]
            v: torch.Tensor = image[..., 2, :, :]

            hi: torch.Tensor = torch.floor(h * 6) % 6
            f: torch.Tensor = ((h * 6) % 6) - hi
            one: torch.Tensor = torch.tensor(1., dtype=image.dtype, device=image.device)
            p: torch.Tensor = v * (one - s)
            q: torch.Tensor = v * (one - f * s)
            t: torch.Tensor = v * (one - (one - f) * s)

            out: torch.Tensor = torch.stack([hi, hi, hi], dim=-3)

            out[out == 0] = torch.stack((v, t, p), dim=-3)[out == 0]
            out[out == 1] = torch.stack((q, v, p), dim=-3)[out == 1]
            out[out == 2] = torch.stack((p, v, t), dim=-3)[out == 2]
            out[out == 3] = torch.stack((p, q, v), dim=-3)[out == 3]
            out[out == 4] = torch.stack((t, p, v), dim=-3)[out == 4]
            out[out == 5] = torch.stack((v, p, q), dim=-3)[out == 5]

            return out
    hsv2rgb = HsvToRgb()
    hsv2rgb.to(device)
    hsv2rgb_scripted = torch.jit.script(hsv2rgb)
    test_out = hsv2rgb_scripted(torch.rand(2, 3, 10, 12, dtype=torch.float32, device=device))
    assert test_out.shape == (2, 3, 10, 12)
    assert test_out.dtype == torch.float32
    torch.jit.save(hsv2rgb_scripted, "hsv2rgb.pt")

    input_rgb = torch.rand(3, 10, 20, dtype=torch.float32, device=device)
    print("Input rgb, min=%f, max=%f"%(torch.min(input_rgb).item(), torch.max(input_rgb).item()))
    input_hsv = rgb2hsv_scripted(input_rgb)
    print("Input hsv, min=%f, max=%f"%(torch.min(input_hsv).item(), torch.max(input_hsv).item()))
    output_rgb = hsv2rgb_scripted(input_hsv)
    error = torch.max(torch.abs(output_rgb-input_rgb)).item()
    print("Error:", error)
    

    print("Done")
