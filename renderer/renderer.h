#pragma once

#include "commons.h"
#include "settings.h"
#include "volume.h"
#include <vector>
#include <tuple>
#include <cuda_runtime.h>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

/**
 * \brief Computes the ambient occlusion parameters.
 * 
 * \param samples the number of AO samples
 * \param rotations the number of random rotations of the kernel in every direction
 * \return a tuple with
 *   - a vector of size 'samples' containing the sample directions on the hemisphere
 *   - a vector of size 'rotations*rotations' containing the random rotations
 */
MY_API std::tuple<std::vector<float4>, std::vector<float4>> computeAmbientOcclusionParameters(int samples, int rotations);

/**
 * Channels:
 * 0: mask
 * 1,2,3: normal x,y,z
 * 4: depth
 * 5: ao
 * 6,7: flow x,y
 */
constexpr int IsoRendererOutputChannels = 8;
/**
 * Channels:
 * 0,1,2: rgb
 * 3: alpha
 * 4,5,6: normal x,y,z
 * 7: depth
 * 8,9: flow x,y
 */
constexpr int DvrRendererOutputChannels = 10;

/**
 * Renders the volume with the specified setting into the output tensor.
 * The rendering is restricted to the viewport specified in the render args.
 * \param volume the volume to render, must reside on the GPU
 * \param args the render arguments
 * \param output the output tensor, a float tensor on the GPU of size BxHxW
 *  with B = 8 (mask, normalX, normalY, normalZ, depth, ao, flowX, flowY)
 *  H = args->viewport.w, W = args->viewport.z
 * \param stream the cuda stream. Common values:
 *  - 0: the global, synchronizing stream
 *  - cuMat::Context::current().stream() for syncing with cuMat
 *    (defined in <cuMat/src/Context.h>)
 *  - at::cuda::getCurrentCUDAStream() for syncing with PyTorch
 *    (defined in <ATen/cuda/CUDAContext.h>)
 */
MY_API void render_gpu(
	const Volume* volume,
	const RendererArgs* args, 
	torch::Tensor& output,
	cudaStream_t stream);

/**
 * \brief Renders the volume with the specified setting at the specified sampling positions.
 * The sampling positions specify the (possibly fractional) location of the ray in screen space.
 * Each input sample position generates an entry in the output tensor.
 * 
 * The rendering is additionally restricted to the viewport specified in the render args.
 * Samples which are outside of the viewport are set to zero in the output
 * (i.e. no intersection)
 * 
 * \param volume the volume to render, must reside on the GPU
 * \param args the render arguments
 * \param sample_positions the sample positions, a float-tensor on the GPU of shape 2xN
 *   where N is the number of samples and the position is then specified as the tuple (x,y)
 * \param samples_out the output tensor, a float tensor on the GPU of size BxN
 *   where N is same number of samples as in 'sample_positions' and
 *   B is the number of channels (see IsoRendererOutputChannels and DvrRendererOutputChannels)
 * \param stream the cuda stream. Common values:
 *  - 0: the global, synchronizing stream
 *  - cuMat::Context::current().stream() for syncing with cuMat
 *    (defined in <cuMat/src/Context.h>)
 *  - at::cuda::getCurrentCUDAStream() for syncing with PyTorch
 *    (defined in <ATen/cuda/CUDAContext.h>)
 */
MY_API void render_samples_gpu(
	const Volume* volume,
	const RendererArgs* args,
	const torch::Tensor& sample_positions,
	torch::Tensor& samples_out,
	cudaStream_t stream);

/**
 * Renders the volume with the specified setting into the output tensor
 * where the stepsize is given per pixel by an input tensor..
 * The rendering is restricted to the viewport specified in the render args.
 * \param volume the volume to render, must reside on the GPU
 * \param args the render arguments
 * \param stepsize the stepsize of the raytracer, a float tensor on the GPU
 *  of size HxW with only positive values.
 * \param output the output tensor, a float tensor on the GPU of size BxHxW
 *  with B = 8 (mask, normalX, normalY, normalZ, depth, ao, flowX, flowY)
 *  H = args->viewport.w, W = args->viewport.z
 * \param stream the cuda stream. Common values:
 *  - 0: the global, synchronizing stream
 *  - cuMat::Context::current().stream() for syncing with cuMat
 *    (defined in <cuMat/src/Context.h>)
 *  - at::cuda::getCurrentCUDAStream() for syncing with PyTorch
 *    (defined in <ATen/cuda/CUDAContext.h>)
 */
MY_API void render_adaptive_stepsize_gpu(
	const Volume* volume,
	const RendererArgs* args,
	const torch::Tensor& stepsize,
	torch::Tensor& output,
	cudaStream_t stream);

/**
 * \brief Scatters the samples to the pixels
 * in 'image_out'.
 * For the specification of 'sample_positions'
 * and 'samples' see \ref render_samples_gpu.
 * Samples that are outside the image are ignored.
 * Pixels in the output images that have no sample
 * are left untouched.
 * \param sample_positions 2*N
 * \param samples Channels*N
 * \param image_out Channels*H*W
 * \param sample_mask_out the output tensor of shape 1xHxW that is '1' at positions
 *   where samples were taken and '0' else.
 */
MY_API void scatter_samples_to_image_gpu(
	const torch::Tensor& sample_positions,
	const torch::Tensor& samples,
	torch::Tensor& image_out,
	torch::Tensor& sample_mask_out,
	cudaStream_t stream);

/**
 * Initializes the renderer.
 * For now this only sets the ambient occlusion sample directions.
 * Returns 1 on success, a negative value on failure
 */
MY_API int64_t initializeRenderer();

/**
 * PyTorch interface: calls <code>render_gpu(TheVolume, TheRendererArgs, newTensorOfMatchingResolution)</code>
 */
MY_API torch::Tensor Render();

/**
 * PyTorch interface: calls
 * <code>render_samples_gpu(TheVolume, TheRendererArgs, sample_position, newTensorOfMatchingResolution)</code>
 */
MY_API torch::Tensor RenderSamples(const torch::Tensor& sample_position);

/**
 * PyTorch interface: calls
 * <code>render_adaptive_stepsize_gpu(TheVolume, TheRendererArgs, stepsizes, newTensorOfMatchingResolution)</code>
 */
MY_API torch::Tensor RenderAdaptiveStepsize(const torch::Tensor& stepsizes);

/**
 * \brief PyTorch interface:
 * scatters the samples to an image
 * \param sample_position the sample positions
 * \param samples the computed samples from \ref RenderSamples
 * \param width the width of the image
 * \param height the height of the image
 * \param default_values the default values for the 8 channels
 * \return the output tensor of shape 8 * Height * Width.
 */
MY_API torch::Tensor ScatterSamplesToImage(
	const torch::Tensor& sample_position, const torch::Tensor& samples,
	int64_t width, int64_t height, std::vector<double> default_values);

END_RENDERER_NAMESPACE
#endif
