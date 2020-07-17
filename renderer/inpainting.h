#pragma once

#include "commons.h"
#include <cuda_runtime.h>

#ifdef RENDERER_HAS_INPAINTING
BEGIN_RENDERER_NAMESPACE

struct MY_API Inpainting
{

	/**
	 * \brief Applies fast inpainting via down- and upsampling
	 * All tensors must reside on the GPU and are of type float or double.
	 *
	 * The mask is defined as:
	 *  - 1: non-empty pixel
	 *  - 0: empty pixel
	 *  To be precise, the input mask is checked by >=0.5.
	 * 
	 * \param mask the mask of shape (Batch, Height, Width)
	 * \param data the data of shape (Batch, Channels, Height, Width)
	 * \return the inpainted data of shape (Batch, Channels, Height, Width)
	 */
	static torch::Tensor fastInpaint(
		const torch::Tensor& mask,
		const torch::Tensor& data);

	/**
	 * \brief Applies fast inpainting via down- and upsampling
	 * with fractional masks. This has a similar effect as the adaptive smoothing.
	 * All tensors must reside on the GPU and are of type float or double.
	 *
	 * The mask is defined as:
	 *  - 1: non-empty pixel
	 *  - 0: empty pixel
	 *  and any fraction in between.
	 *
	 * \param mask the mask of shape (Batch, Height, Width)
	 * \param data the data of shape (Batch, Channels, Height, Width)
	 * \return the inpainted data of shape (Batch, Channels, Height, Width)
	 */
	static torch::Tensor fastInpaintFractional(
		const torch::Tensor& mask,
		const torch::Tensor& data);

	/**
	 * \brief Adjoint code for \ref fastInpaintFractional.
	 * \param mask the input mask of shape (Batch, Height, Width)
	 * \param data the input data of shape (Batch, Channels, Height, Width)
	 * \param gradOutput the gradient of the output of shape (Batch, Channels, Height, Width)
	 * \param outGradMask the gradient of the input mask of shape (Batch, Height, Width),
	 *		should be initialized with zero
	 * \param outGradData the gradient of the input data of shape (Batch, Channels, Height, Width),
	 *		should be initialized with zero
	 * \return tuple:
	 *    - outGradMask the gradient of the input mask of shape (Batch, Height, Width),
	 *    - outGradData the gradient of the input data of shape (Batch, Channels, Height, Width),
	 */
	static std::tuple<torch::Tensor, torch::Tensor>
	adjFastInpaintingFractional(
		const torch::Tensor& mask,
		const torch::Tensor& data,
		const torch::Tensor& gradOutput);

	/**
	 * \brief Applies fast inpainting via down- and upsampling
	 * All tensors must reside on the GPU and are of type float or double.
	 *
	 * The mask is defined as:
	 *  - 1: non-empty pixel
	 *  - 0: empty pixel
	 *  To be precise, the input mask is checked by >=0.5.
	 *
	 * \param mask the mask of shape (Batch, Height, Width)
	 * \param data the data of shape (Batch, Channels, Height, Width)
	 * \param iterations [Out] the number of iterations on convergence
	 * \param residual [Out] the final residual norm
	 * \param m0 the maximal number of iterations of the whole optimization
	 * \param epsilon early termination if the error falls below this threshold
	 * \param m1 number of pre-smoothing steps
	 * \param m2 number of post-smoothing steps
	 * \param mc cycling strategy, mc=1->V-cycle, mc=2->W-cycle
	 * \param ms grid size after which the equation is directly solved.
	 *   If ms==-1,the largest size that fits into a CUDA block is used.
	 * \param m3 maximal iteration for the coarsest grid
	 * \return the inpainted data of shape (Batch, Channels, Height, Width)
	 */
	static torch::Tensor pdeInpaint(
		const torch::Tensor& mask,
		const torch::Tensor& data,
		int& iterations, float& residual,
		int64_t m0 = 50, double epsilon = 1e-2,
		int64_t m1 = 3, int64_t m2 = 1,
		int64_t mc = 2, int64_t ms = -1, int64_t m3 = 20);
};

END_RENDERER_NAMESPACE
#endif
