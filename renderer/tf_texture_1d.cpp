#include "tf_texture_1d.h"

#include <cuMat/src/Context.h>
#include <algorithm>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

TfTexture1D::TfTexture1D(int size)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //RGBA

	CUMAT_SAFE_CALL(cudaMallocArray(&gpuData_.cudaArray_, &channelDesc, size, 0, cudaArraySurfaceLoadStore));
	gpuData_.cudaArraySize_ = size;

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = gpuData_.cudaArray_;

	//Create the surface object.
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&gpuData_.surfaceObject_, &resDesc));

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeClamp);
	texDesc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	texDesc.normalizedCoords = 1;

	//Create the texture object.
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&gpuData_.textureObject_, &resDesc, &texDesc, nullptr));
}

TfTexture1D::~TfTexture1D()
{
	destroy();
	CUMAT_SAFE_CALL(cudaDestroyTextureObject(gpuData_.textureObject_));
	CUMAT_SAFE_CALL(cudaDestroySurfaceObject(gpuData_.surfaceObject_));
	CUMAT_SAFE_CALL(cudaFreeArray(gpuData_.cudaArray_));
}

bool TfTexture1D::updateIfChanged(const std::vector<float>& densityAxisOpacity, const std::vector<float>& opacityAxis,
	const std::vector<float>& densityAxisColor, const std::vector<float3>& colorAxis)
{
	//First check if we have the same values for axes and values. If so, do not allocate new memory or create texture object again. 

	//std::vector cannot compare const vs non-const vectors. I don't want to use const_cast here.
	//Also, float3 has an equal operator which returns element-wise boolean. Since I cannot overload it, std::equal is the best option.
	bool changed = densityAxisOpacity.size() != densityAxisOpacity_.size() ||
		opacityAxis.size() != opacityAxis_.size() ||
		densityAxisColor.size() != densityAxisColor_.size() ||
		colorAxis.size() != colorAxis_.size();

	changed = changed ||
		!std::equal(densityAxisOpacity.cbegin(), densityAxisOpacity.cend(), densityAxisOpacity_.cbegin()) ||
		!std::equal(opacityAxis.cbegin(), opacityAxis.cend(), opacityAxis_.cbegin()) ||
		!std::equal(densityAxisColor.cbegin(), densityAxisColor.cend(), densityAxisColor_.cbegin()) ||
		!std::equal(colorAxis.cbegin(), colorAxis.cend(), colorAxis_.cbegin(), [](const float3& l, const float3& r)
			{
				return l.x == r.x && l.y == r.y && l.z == r.z;
			});

	if (changed)
	{
		destroy();

		gpuData_.sizeOpacity_ = densityAxisOpacity.size();
		assert(gpuData_.sizeOpacity_ == opacityAxis.size());
		assert(gpuData_.sizeOpacity_ >= 1);

		gpuData_.sizeColor_ = densityAxisColor.size();
		assert(gpuData_.sizeColor_ == colorAxis.size());
		assert(gpuData_.sizeColor_ >= 1);

		//Transfer from host to device
		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.densityAxisOpacity_, gpuData_.sizeOpacity_ * sizeof(float)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.densityAxisOpacity_, densityAxisOpacity.data(), gpuData_.sizeOpacity_ * sizeof(float), cudaMemcpyHostToDevice));

		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.opacityAxis_, gpuData_.sizeOpacity_ * sizeof(float)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.opacityAxis_, opacityAxis.data(), gpuData_.sizeOpacity_ * sizeof(float), cudaMemcpyHostToDevice));

		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.densityAxisColor_, gpuData_.sizeColor_ * sizeof(float)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.densityAxisColor_, densityAxisColor.data(), gpuData_.sizeColor_ * sizeof(float), cudaMemcpyHostToDevice));

		CUMAT_SAFE_CALL(cudaMalloc(&gpuData_.colorAxis_, gpuData_.sizeColor_ * sizeof(float3)));
		CUMAT_SAFE_CALL(cudaMemcpy(gpuData_.colorAxis_, colorAxis.data(), gpuData_.sizeColor_ * sizeof(float3), cudaMemcpyHostToDevice));

		densityAxisOpacity_ = densityAxisOpacity;
		opacityAxis_ = opacityAxis;
		densityAxisColor_ = densityAxisColor;
		colorAxis_ = colorAxis;

		computeCudaTexture(gpuData_);
	}

	return changed;
}

void TfTexture1D::destroy()
{
	if (gpuData_.densityAxisOpacity_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.densityAxisOpacity_));
		gpuData_.densityAxisOpacity_ = nullptr;
	}
	if (gpuData_.opacityAxis_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.opacityAxis_));
		gpuData_.opacityAxis_ = nullptr;
	}
	if (gpuData_.densityAxisColor_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.densityAxisColor_));
		gpuData_.densityAxisColor_ = nullptr;
	}
	if (gpuData_.colorAxis_)
	{
		CUMAT_SAFE_CALL(cudaFree(gpuData_.colorAxis_));
		gpuData_.colorAxis_ = nullptr;
	}
	gpuData_.sizeOpacity_ = 0;
	gpuData_.sizeColor_ = 0;

	densityAxisOpacity_.clear();
	opacityAxis_.clear();
	densityAxisColor_.clear();
	colorAxis_.clear();
}

END_RENDERER_NAMESPACE
#endif
