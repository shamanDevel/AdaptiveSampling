#include "tf_texture_1d.h"

#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

__device__ float4 fetch(float density, const TfTexture1D::GpuData& gpuData)
{
	float4 result{ 0.0f, 0.0f, 0.0f, 0.0f };

	//This can be done using binary search but now simply do linear search.
	for (int i = 0; i < gpuData.sizeColor_ + 1; ++i)
	{
		auto leftDensity = i == 0 ? 0.0f : gpuData.densityAxisColor_[i - 1];
		auto rightDensity = i == gpuData.sizeColor_ ? 1.0f : gpuData.densityAxisColor_[i];
		if (density < rightDensity)
		{
			auto leftColor = i == 0 ? gpuData.colorAxis_[0] : gpuData.colorAxis_[i - 1];
			auto rightColor = i == gpuData.sizeColor_ ? gpuData.colorAxis_[gpuData.sizeColor_ - 1] : gpuData.colorAxis_[i];

			auto t = (density - leftDensity) / (rightDensity - leftDensity);

			auto rgb = labToRgb(lerp(leftColor, rightColor, t));
			result.x = rgb.x;
			result.y = rgb.y;
			result.z = rgb.z;
			break;
		}
	}
	for (int i = 0; i < gpuData.sizeOpacity_ + 1; ++i)
	{
		auto leftDensity = i == 0 ? 0.0f : gpuData.densityAxisOpacity_[i - 1];
		auto rightDensity = i == gpuData.sizeOpacity_ ? 1.0f : gpuData.densityAxisOpacity_[i];
		if (density < rightDensity)
		{
			auto leftOpacity = i == 0 ? gpuData.opacityAxis_[0] : gpuData.opacityAxis_[i - 1];
			auto rightOpacity = i == gpuData.sizeOpacity_ ? gpuData.opacityAxis_[gpuData.sizeOpacity_ - 1] : gpuData.opacityAxis_[i];

			auto t = (density - leftDensity) / (rightDensity - leftDensity);

			result.w = lerp(leftOpacity, rightOpacity, t);
			break;
		}
	}

	return result;
}

__global__ void ComputeCudaTextureKernel(
	dim3 virtualSize,
	TfTexture1D::GpuData gpuData)
{
	CUMAT_KERNEL_1D_LOOP(x, virtualSize)

	auto density = x / static_cast<float>(virtualSize.x);
	auto rgbo = fetch(density, gpuData);

	surf1Dwrite(rgbo.x, gpuData.surfaceObject_, x * 16);
	surf1Dwrite(rgbo.y, gpuData.surfaceObject_, x * 16 + 4);
	surf1Dwrite(rgbo.z, gpuData.surfaceObject_, x * 16 + 8);
	surf1Dwrite(rgbo.w, gpuData.surfaceObject_, x * 16 + 12);

	CUMAT_KERNEL_1D_LOOP_END
}

MY_API void computeCudaTexture(const TfTexture1D::GpuData& gpuData)
{
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(gpuData.cudaArraySize_, ComputeCudaTextureKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	ComputeCudaTextureKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, gpuData);
	CUMAT_CHECK_ERROR();
}

END_RENDERER_NAMESPACE
#endif