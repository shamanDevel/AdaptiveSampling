#pragma once

#include "commons.h"
#include "helper_math.h"

#include <cuda_runtime.h>
#include <vector>

#ifdef RENDERER_HAS_RENDERER

BEGIN_RENDERER_NAMESPACE

//From https://github.com/berendeanicolae/ColorSpace/blob/master/src/Conversion.cpp
//Assumes color channels are in [0,1]
#ifdef __NVCC__
__device__ __host__
#endif
inline float3 rgbToXyz(const float3& rgb)
{
	auto r = ((rgb.x > 0.04045f) ? pow((rgb.x + 0.055f) / 1.055f, 2.4f) : (rgb.x / 12.92f)) * 100.0f;
	auto g = ((rgb.y > 0.04045f) ? pow((rgb.y + 0.055f) / 1.055f, 2.4f) : (rgb.y / 12.92f)) * 100.0f;
	auto b = ((rgb.z > 0.04045f) ? pow((rgb.z + 0.055f) / 1.055f, 2.4f) : (rgb.z / 12.92f)) * 100.0f;

	auto x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
	auto y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
	auto z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

	return { x, y, z };
}

//Output color channels are in [0,1]
#ifdef __NVCC__
__device__ __host__
#endif
inline float3 xyzToRgb(const float3& xyz)
{
	auto x = xyz.x / 100.0f;
	auto y = xyz.y / 100.0f;
	auto z = xyz.z / 100.0f;

	auto r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
	auto g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
	auto b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

	r = (r > 0.0031308f) ? (1.055f * pow(r, 1.0f / 2.4f) - 0.055f) : (12.92f * r);
	g = (g > 0.0031308f) ? (1.055f * pow(g, 1.0f / 2.4f) - 0.055f) : (12.92f * g);
	b = (b > 0.0031308f) ? (1.055f * pow(b, 1.0f / 2.4f) - 0.055f) : (12.92f * b);

	return { r, g, b };
}

#ifdef __NVCC__
__device__ __host__
#endif
inline float3 rgbToLab(const float3& rgb)
{
	auto xyz = rgbToXyz(rgb);

	auto x = xyz.x / 95.047f;
	auto y = xyz.y / 100.00f;
	auto z = xyz.z / 108.883f;

	x = (x > 0.008856f) ? cbrt(x) : (7.787f * x + 16.0f / 116.0f);
	y = (y > 0.008856f) ? cbrt(y) : (7.787f * y + 16.0f / 116.0f);
	z = (z > 0.008856f) ? cbrt(z) : (7.787f * z + 16.0f / 116.0f);

	return { (116.0f * y) - 16.0f, 500.0f * (x - y), 200.0f * (y - z) };
}

#ifdef __NVCC__
__device__ __host__
#endif
inline float3 labToRgb(const float3& lab)
{
	auto y = (lab.x + 16.0f) / 116.0f;
	auto x = lab.y / 500.0f + y;
	auto z = y - lab.z / 200.0f;

	auto x3 = x * x * x;
	auto y3 = y * y * y;
	auto z3 = z * z * z;

	x = ((x3 > 0.008856f) ? x3 : ((x - 16.0f / 116.0f) / 7.787f)) * 95.047f;
	y = ((y3 > 0.008856f) ? y3 : ((y - 16.0f / 116.0f) / 7.787f)) * 100.0f;
	z = ((z3 > 0.008856f) ? z3 : ((z - 16.0f / 116.0f) / 7.787f)) * 108.883f;

	return xyzToRgb({ x, y, z });
}

//This class should be instantiated by calling static "allocate" function.
//It doesn't have an explicit destructor definition so that it can be directly copied into a CUDA kernel as argument.
//Do not forget to call static function "free" after you're done with texture.
class MY_API TfTexture1D
{
public:
	struct GpuData
	{
		int sizeOpacity_{ 0 };
		float* densityAxisOpacity_{ nullptr };
		float* opacityAxis_{ nullptr };

		int sizeColor_{ 0 };
		float* densityAxisColor_{ nullptr };
		float3* colorAxis_{ nullptr };

		cudaArray_t cudaArray_{ nullptr };
		int cudaArraySize_{ 0 };
		cudaSurfaceObject_t surfaceObject_{ 0 };
		cudaTextureObject_t textureObject_{ 0 };
	};

public:
	TfTexture1D(int size = 512);
	TfTexture1D(TfTexture1D&&) = delete;
	TfTexture1D(const TfTexture1D&) = delete;
	~TfTexture1D();

	//This function expects colors in CIELab space and TfTexture1D acts accordingly.
	bool updateIfChanged(const std::vector<float>& densityAxisOpacity, const std::vector<float>& opacityAxis,
		const std::vector<float>& densityAxisColor, const std::vector<float3>& colorAxis);

	cudaTextureObject_t getTextureObject() const { return gpuData_.textureObject_; }

private:
	GpuData gpuData_;
	std::vector<float> densityAxisOpacity_;
	std::vector<float> opacityAxis_;
	std::vector<float> densityAxisColor_;
	std::vector<float3> colorAxis_;

private:
	void destroy();
};

MY_API void computeCudaTexture(const TfTexture1D::GpuData& gpuData);

END_RENDERER_NAMESPACE
#endif