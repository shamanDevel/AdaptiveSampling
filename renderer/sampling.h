#pragma once

#include "commons.h"
#include <cuda_runtime.h>
#include <vector>

#ifdef RENDERER_HAS_SAMPLING
BEGIN_RENDERER_NAMESPACE

/**
 * \brief Load sample positions from file.
 * Currently, only plain .txt files are supported
 * \param filename the filename
 * \return a tuple with:
 *  - first: the sample positions as a 2*N tensor on the CPU (!)
 *  - second: an M*3 int-tensor on the CPU with the triangle indices
 *  - third: the width of the grid where the points were created on
 *  - forth: the height of the grid where the points were created on
 */
MY_API std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>
loadSamplesFromFile(const std::string& filename);

/**
 * \brief Computes the average distance between samples on the CPU
 * \param samplePositions the sample positions as a 2*N float tensor on the CPU
 * \param sampleIndices an M*3 int-tensor with the triangle indices on the CPU
 * \return a 1D float tensor of length N with the sample distance per sample
 */
MY_API torch::Tensor computeAverageSampleDistance(
	const torch::Tensor& samplePositions,
	const torch::Tensor& sampleIndices);

/**
 * \brief Performs an adaptive smoothing on the GPU.
 * \param
 *	input the input tensor of shape B*C*H*W of type float on the gpu.
 *	This tensor will be smoothed based on the smoothing radius defined in 'distances'
 * \param distances
 *	a tensor of shape B*1*H*W defining the smoothing radius in pixels.
 * \param distanceToStandardDeviation
 *	a scaling factor to convert the distance value into the standard deviation
 *	for the gaussian kernel.
 *	A good value is <br>0.5</br>, leading to around 95% of the gaussian kernel
 *	falls within the sampling radius.
 * \return 
 */
MY_API torch::Tensor adaptiveSmoothing(
	const torch::Tensor& input,
	const torch::Tensor& distances,
	double distanceToStandardDeviation);

/**
 * \brief Adjoint / Backprop of the adaptive smoothing function.
 * \param input the input tensor
 * \param distances the distances tensor
 * \param distanceToStandardDeviation the standard deviation scaling factor
 * \param gradOutput the gradient of the output to be backpropagated
 * \param writeGradInput true if out_gradInput should be filled
 * \param writeGradDistances true if out_gradDistances should be filled
 * \return tuple:
      - out_gradInput: the gradient with respect to the input tensor
      - out_gradDistances: the gradient with respect to the distances tensor
 */
MY_API std::tuple<torch::Tensor, torch::Tensor> adjAdaptiveSmoothing(
	const torch::Tensor& input,
	const torch::Tensor& distances,
	double distanceToStandardDeviation,
	const torch::Tensor& gradOutput,
	bool writeGradInput,
	bool writeGradDistances);

END_RENDERER_NAMESPACE
#endif
