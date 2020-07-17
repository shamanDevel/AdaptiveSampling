import numpy as np
import torch

class Multigrid2D:
    """
    Multigrid for the 2D Laplace equation.
    """

    def __next_power_of_two(self, x):
        return 1<<(x-1).bit_length()

    def __init__(self, size : int, h : float, 
                 m0 : int = 50, epsilon : float = 1e-2, 
                 m1 : int = 3, m2 : int = 1, 
                 mc : int = 2, ms : int = 5, m3 : int = 20):
        """
        Initializes the multigrid solver for 2D grids of size 'size'.

        Parameters:
        size -- the grid size
        h -- the grid spacing
        m0 -- the maximal number of iterations of the whole optimization
        epsilon -- early termination if the error falls below this threshold
        m1 -- number of presmoothing steps
        m2 -- number of postsmoothign steps
        mc -- cycling strategy, mc=1->V-cycle, mc=2->W-cycle
        ms -- grid size after which the equation is directly solved
        m3 -- maximal iteration for the coarsest grid
        """

        self._size = self.__next_power_of_two(size-1) + 1
        self._h = h
        self._m0 = m0
        self._epsilon = epsilon
        self._m1 = m1
        self._m2 = m2
        self._mc = mc
        self._ms = ms
        self._m3 = m3
        self._epsilonCoarse = 1e-5

    def __smooth(self, u, f, mask, h : float):
        """
        Performs one iteration of a Jaobi smoother.
        All variables are 2D arrays of the same size
        Parameters:
          u^k: current solution (C, size, size)
          f: right hand side (C, size, size)
          mask: boolean array where Dirichlet Boundaries are applied (size, size)
          h: grid spacing
        Returns:
          u^(k+1)
        """
        s = u.shape
        # the four neighbors
        n1 = np.copy(u); n1[...,:,1:] = u[...,:,:-1]
        n2 = np.copy(u); n2[...,:,:-1] = u[...,:,1:]
        n3 = np.copy(u); n3[...,1:,:] = u[...,:-1,:]
        n4 = np.copy(u); n4[...,:-1,:] = u[...,1:,:]
        # assemble non-boundary entries
        newU = (-h/4) * (f - (1/h)*(n1+n2+n3+n4))
        # select / overwrite dirichlet boundaries
        newU[mask] = f[mask]
        return newU

    def __computeResidual(self, u, f, mask, h : float):
        """
        Comptues the residual f-Au
        All variables are 2D arrays of the same size
        Parameters:
          u^k: current solution
          f: right hand side
          mask: boolean array where Dirichlet Boundaries are applied
          h: grid spacing
        Returns:
          the residual f-Au
        """
        s = u.shape
        # the four neighbors
        n1 = np.copy(u); n1[...,:,1:] = u[...,:,:-1]
        n2 = np.copy(u); n2[...,:,:-1] = u[...,:,1:]
        n3 = np.copy(u); n3[...,1:,:] = u[...,:-1,:]
        n4 = np.copy(u); n4[...,:-1,:] = u[...,1:,:]
        # compute f-Au
        r = f - (1/h)*(n1+n2+n3+n4-4*u)
        # cope with dirichlet boundaries
        r[mask] = f[mask] - u[mask]
        return r

    def __restrict(self, f, mask):
        """
        Restricts the residual / new right-hand-side f onto the coarser grid.
        This method assumes, that the size of f is a power-of-two plus 1, hence contains a power-of-two intervals.
        """
        s = f.shape
        sSmall = (s[0], (s[1]-1)//2 + 1, (s[2]-1)//2 + 1)

        fSmall = np.zeros(tuple(list(sSmall)+[3]), dtype=f.dtype)
        fInput = np.stack([f, mask*1.0, np.ones_like(f)], axis=3)

        #        [ 1 2 1 ]
        # 1/16 * [ 2 4 2 ]
        #        [ 1 2 1 ]
        fSmall += 4 * fInput[..., ::2, ::2, :]
        fSmall[..., 1:, :, :] += 2 * fInput[..., 1::2, ::2, :]
        fSmall[..., :-1, :, :] += 2 * fInput[..., 1::2, ::2, :]
        fSmall[..., :, 1:, :] += 2 * fInput[..., ::2, 1::2, :]
        fSmall[..., :, :-1, :] += 2 * fInput[..., ::2, 1::2, :]
        fSmall[..., 1:, 1:, :] += fInput[..., 1::2, 1::2, :]
        fSmall[..., 1:, :-1, :] += fInput[..., 1::2, 1::2, :]
        fSmall[..., :-1, 1:, :] += fInput[..., 1::2, 1::2, :]
        fSmall[..., :-1, :-1, :] += fInput[..., 1::2, 1::2, :]

        vSmall = fSmall[..., :,:, 0] / fSmall[..., :,:, 2]         # values
        mSmall = (fSmall[..., :,:, 1] / fSmall[..., :,:, 2]) > 0.1 # mask
        vSmall[mSmall] = np.zeros_like(vSmall[mSmall]) #vSmall[mSmall].fill(0) # residual at the boundary is zero
        return vSmall, mSmall

    def __interpolate(self, v, mask):
        """
        Interpolates the coarse grid correction onto the finer grid.
        """
        s = v.shape
        sLarge = (s[0], (s[1]-1)*2 + 1, (s[2]-1)*2 + 1)

        vLarge = np.zeros(sLarge, dtype=v.dtype)
        #       ] 1 2 1 [
        # 1/4 * ] 2 4 2 [
        #       ] 1 2 1 [
        vLarge[..., ::2, ::2] = v[..., ::1, ::1]
        vLarge[..., 1::2, ::2] = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
        vLarge[..., ::2, 1::2] = 0.5 * (v[..., :, :-1] + v[..., :, 1:])
        vLarge[..., 1::2, 1::2] = 0.25 * (v[..., :-1, :-1] + v[..., 1:, :-1] + v[..., :-1, 1:] + v[..., 1:, 1:])
        vLarge[mask] = np.zeros_like(vLarge[mask]) #vLarge[mask].fill(0)
        return vLarge

    def __multigrid(self, u, f, mask, h):
        size = u.shape[1]
        if size <= self._ms:
            # solve coarse grid equation
            i = 0
            for i in range(self._m3):
                i += 1
                u = self.__smooth(u, f, mask, h)
                error = np.linalg.norm(self.__computeResidual(u, f, mask, h))
                if error<=self._epsilonCoarse:
                    #print("Coarse grid converged after", i, "iterations with an error of", error)
                    break
            #else:
            #    print("Coarse grid did not converge after", i, "iterations, error left is", error)
            return u

        # presmooth
        for i in range(self._m1):
            u = self.__smooth(u, f, mask, h)

        # compute residual
        r = self.__computeResidual(u, f, mask, h)

        # restrict to coarse grid
        fCoarse, maskCoarse = self.__restrict(r, mask)
        hCoarse = h * 2

        # recursion
        vCoarse = np.zeros_like(fCoarse)
        for i in range(self._mc):
            vCoarse = self.__multigrid(vCoarse, fCoarse, maskCoarse, hCoarse)

        # interpolate
        v = self.__interpolate(vCoarse, mask)

        # correct
        uNew = u + v

        # post-smooth
        for i in range(self._m2):
            uNew = self.__smooth(uNew, f, mask, h)

        # done
        return uNew

    def solve(self, mask, boundary):
        """
        Solves the 2D-laplace problem with Dirichlet boundaries
        on the inside of the grid and Neumann boundaries on all sides.

        mask -- boolean array with True where 'boundary' contains the 
                fixed Dirichlet boundaries.
                Shape: (1, size, size)
        boundary -- the dirichlet boudary values
                Shape: (channels, size, size)
        """

        mask = np.concatenate([mask]*boundary.shape[0], axis=0).astype(bool)
        boundary = np.where(
            mask, 
            boundary,
            np.zeros_like(boundary))
        
        # pad to size self._size
        channels, sizeW, sizeH = boundary.shape
        pad = ((0, 0), (0, self._size - sizeW), (0, self._size - sizeH))
        mask = np.pad(mask, pad)
        boundary = np.pad(boundary, pad)

        # run multigrid
        u = np.zeros_like(boundary)
        for i in range(self._m0):
            u = self.__multigrid(u, boundary, mask, self._h)
            error = np.linalg.norm(self.__computeResidual(u, boundary, mask, self._h))
            #print("Iteration", (i+1), ", error:", error)
            if error < self._epsilon:
                break

        # done, remove pad
        return u[..., :sizeW, :sizeH]

def pdeInpaint(input, sample_mask, cpu = False):
    if cpu:
        B, C, H, W = input.shape
        m = Multigrid2D(max(H, W), 1/max(H, W), m0=50)
        output = [
            torch.from_numpy(
                m.solve(sample_mask[b, ...].cpu().numpy(), input[b, ...].cpu().numpy())) \
                    .to(device=input.device, dtype=input.dtype)
            for b in range(B)]
        return torch.stack(output, dim=0)
    else:
        return torch.ops.renderer.pde_inpaint(
                sample_mask[:,0,:,:], # remove channel
                input, 
                200, 1e-4, 5, 2, # m0, epsilon, m1, m2
                0, # mc -> multigrid recursion count. =0 disables the multigrid hierarchy
                9, 0) # ms, m3

if __name__ == "__main__":
    """
    Test how the pde impainting works and compare it to the fast inpainting
    """

    from importance.gradientMap import GradientImportanceMap
    from importance.importanceMap import ImportanceMap
    from sampling.uniform import UniformSampler
    import imageio
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    torch.ops.load_library("./Renderer.dll")

    # load image
    filename = "screenshots/dense1_png/rm_gt_normal.png"
    img = imageio.imread(filename, pilmode='RGB') / 255.0
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    img = img.to(device=torch.device('cuda'), dtype=torch.float32)
    img = torch.nn.functional.interpolate(img, scale_factor=0.1, mode='area')
    print("Image loaded:", img.shape)
    B, C, H, W = img.shape

    # compute importance
    importance_map = GradientImportanceMap(1, (0,1), (1,1), (2,1))(img)
    min_importance = 0.01
    mean_importance = 0.2
    importance_map = ImportanceMap.normalize(
        importance_map, min_importance, mean_importance, 0)
    importance_map = importance_map.unsqueeze(1)
    print("Importance map computed")

    # generate sampling pattern
    sample_pattern = UniformSampler().fill_image((1, 1, H, W))
    sample_pattern = torch.from_numpy(sample_pattern) \
        .to(device=torch.device('cuda'), dtype=torch.float32) \
        / (H*W)
    print("Sample pattern created")

    # compute samples
    sample_mask = (importance_map >= sample_pattern).to(dtype=importance_map.dtype)
    scaled_image = sample_mask * img

    # perform inpainting
    inpainted_image_pde = pdeInpaint(scaled_image, sample_mask, cpu=False)
    print("Image inpainted with PDE algorithm")
    inpainted_image_fast = torch.ops.renderer.fast_inpaint(
                sample_mask[0], img)

    # visualize sampling + reconstruction
    fig, axes = plt.subplots(ncols=2, nrows=2)
    f1 = axes[0,0].imshow(img[0].permute(1,2,0).cpu().numpy()) #default
    f2 = axes[0,1].imshow(scaled_image[0].permute(1,2,0).cpu().numpy())
    f3 = axes[1,0].imshow(inpainted_image_pde[0].permute(1,2,0).cpu().numpy())
    f4 = axes[1,1].imshow(inpainted_image_fast[0].permute(1,2,0).cpu().numpy())
    axes[0,0].set_title("Input")
    axes[0,1].set_title("Sampling")
    axes[1,0].set_title("PDE Interpolation")
    axes[1,1].set_title("Fast Interpolation")

    fig.suptitle("Inpainting test")
    plt.show()