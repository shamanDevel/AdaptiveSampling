#include "volume.h"

#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

__global__ void ExtractMinMaxKernel(
	dim3 virtualSize,
	cudaTextureObject_t volumeTexture,
	RENDERER_NAMESPACE Volume::Histogram* histogram)
{
	CUMAT_KERNEL_3D_LOOP(x, y, z, virtualSize)

		auto density = tex3D<float>(volumeTexture, x, y, z);

	if (density > 0.0f)
	{
		atomicInc(&histogram->numOfNonzeroVoxels, UINT32_MAX);
	}

	//Since atomicMin and atomicMax does not work with floating point values, trick below can be used.
	//Comparing two non-negative floating point numbers is the same as comparing them as if they are integers.
	atomicMin(reinterpret_cast<int*>(&histogram->minDensity), __float_as_int(density));
	atomicMax(reinterpret_cast<int*>(&histogram->maxDensity), __float_as_int(density));

	CUMAT_KERNEL_3D_LOOP_END
}

__global__ void ExtractHistogramKernel(
	dim3 virtualSize,
	cudaTextureObject_t volumeTexture,
	RENDERER_NAMESPACE Volume::Histogram* histogram,
	int numOfBins)
{
	CUMAT_KERNEL_3D_LOOP(x, y, z, virtualSize)

		auto density = tex3D<float>(volumeTexture, x, y, z);
	if (density > 0.0f)
	{
		auto densityWidthResolution = (histogram->maxDensity - histogram->minDensity) / numOfBins;

		auto binIdx = static_cast<int>((density - histogram->minDensity) / densityWidthResolution);

		//Precaution against floating-point errors
		binIdx = binIdx >= numOfBins ? (numOfBins - 1) : binIdx;
		//atomicInc(reinterpret_cast<unsigned int*>(histogram->bins + binIdx, UINT32_MAX));
		atomicAdd(histogram->bins + binIdx, 1.0f / histogram->numOfNonzeroVoxels);
	}

	CUMAT_KERNEL_3D_LOOP_END
}

Volume::Histogram Volume::extractHistogram() const
{
	Volume::Histogram histogram;
	auto data = getLevel(0);

	Volume::Histogram* histogramGpu;
	CUMAT_SAFE_CALL(cudaMalloc(&histogramGpu, sizeof(Volume::Histogram)));
	CUMAT_SAFE_CALL(cudaMemcpy(histogramGpu, &histogram, sizeof(Volume::Histogram), cudaMemcpyHostToDevice));

	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(data->sizeX(), data->sizeY(), data->sizeZ(), ExtractMinMaxKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	ExtractMinMaxKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, data->dataTexGpu(), histogramGpu);
	CUMAT_CHECK_ERROR();

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	cfg = ctx.createLaunchConfig3D(data->sizeX(), data->sizeY(), data->sizeZ(), ExtractHistogramKernel);
	stream = at::cuda::getCurrentCUDAStream();
	ExtractHistogramKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, data->dataTexGpu(), histogramGpu, histogram.getNumOfBins());
	CUMAT_CHECK_ERROR();

	CUMAT_SAFE_CALL(cudaMemcpy(&histogram, histogramGpu, sizeof(Volume::Histogram), cudaMemcpyDeviceToHost));
	CUMAT_SAFE_CALL(cudaFree(histogramGpu));

	histogram.maxFractionVal = *std::max_element(std::begin(histogram.bins), std::end(histogram.bins));

	return histogram;
}

END_RENDERER_NAMESPACE
#endif
