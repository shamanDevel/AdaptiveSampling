#include "renderer.h"
#include <random>
#include <iomanip>
#include <cuMat/src/Errors.h>
#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include "helper_math.h"
#include "camera.h"
#include "tf_texture_1d.h"

#ifdef RENDERER_HAS_RENDERER

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM((x.dtype() == at::kFloat), #x " must be a float tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x);

namespace std
{
	std::ostream& operator<<(std::ostream& o, const float3& v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const float4& v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z << "," << v.w;
		return o;
	}
}

BEGIN_RENDERER_NAMESPACE

//=========================================
// HELPER 
//=========================================

std::tuple<std::vector<float4>, std::vector<float4>> computeAmbientOcclusionParameters(int samples, int rotations)
{
	static std::default_random_engine rnd;
	static std::uniform_real_distribution<float> distr(0.0f, 1.0f);
	//samples
	std::vector<float4> aoHemisphere(samples);
	for (int i = 0; i < samples; ++i)
	{
		float u1 = distr(rnd);
		float u2 = distr(rnd);
		float r = std::sqrt(u1);
		float theta = 2 * M_PI * u2;
		float x = r * std::cos(theta);
		float y = r * std::sin(theta);
		float scale = distr(rnd);
		scale = 0.1 + 0.9 * scale * scale;
		aoHemisphere[i] = make_float4(x*scale, y*scale, std::sqrt(1 - u1)*scale, 0);
	}
	//random rotation vectors
	std::vector<float4> aoRandomRotations(rotations*rotations);
	for (int i = 0; i < rotations*rotations; ++i)
	{
		float x = distr(rnd) * 2 - 1;
		float y = distr(rnd) * 2 - 1;
		float linv = 1.0f / sqrt(x*x + y * y);
		aoRandomRotations[i] = make_float4(x*linv, y*linv, 0, 0);
	}

	return std::make_tuple(aoHemisphere, aoRandomRotations);
}

//=========================================
// RENDERER CONFIGURATION
//=========================================

#define MAX_AMBIENT_OCCLUSION_SAMPLES 512
__constant__ float4 aoHemisphere[MAX_AMBIENT_OCCLUSION_SAMPLES];
//#define AMBIENT_OCCLUSION_RANDOM_ROTATIONS 4
#define AMBIENT_OCCLUSION_RANDOM_ROTATIONS 1
__constant__ float4 aoRandomRotations[AMBIENT_OCCLUSION_RANDOM_ROTATIONS * AMBIENT_OCCLUSION_RANDOM_ROTATIONS];

struct RendererDeviceSettings
{
	float2 screenSize;
	float3 volumeSize;
	int binarySearchSteps;
	float stepsize;
	float normalStepSize;
	float3 eyePos;
	float4 currentViewMatrixInverse[4]; //row-by-row
	float4 currentViewMatrix[4];
	float4 nextViewMatrix[4];
	float4 normalMatrix[4];
	float3 boxMin;
	float3 boxSize;
	int aoSamples;
	float aoRadius;
	float aoBias;
	int4 viewport; //start x, start y, end x, end y
	float isovalue;
	float opacityScaling{ 1.0f };
	float minDensity{ 0.0f };
	float maxDensity{ 1.0f };
	bool useShading = false;
	ShadingSettings shading;
};

//texture<float, 3, cudaReadModeElementType> float_tex;
//texture<unsigned char, 3, cudaReadModeNormalizedFloat> char_tex;
//texture<unsigned short, 3, cudaReadModeNormalizedFloat> short_tex;

//=========================================
// RENDERER KERNEL
//=========================================

inline __device__ float4 matmul(const float4 mat[4], float4 v)
{
	return make_float4(
		dot(mat[0], v),
		dot(mat[1], v),
		dot(mat[2], v),
		dot(mat[3], v)
	);
}

__device__ inline void writeOutputIso(
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t>& output, int x, int y, 
	float mask = 0, float3 normal = { 0,0,0 }, float depth = 0, float ao = 1, float2 flow = {0,0})
{
	output[0][y][x] = mask;
	output[1][y][x] = normal.x;
	output[2][y][x] = normal.y;
	output[3][y][x] = normal.z;
	output[4][y][x] = depth;
	output[5][y][x] = ao;
	output[6][y][x] = flow.x;
	output[7][y][x] = flow.y;
}

__device__ inline void writeOutputSamplesIso(
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t>& output, int idx,
	float mask = 0, float3 normal = { 0,0,0 }, float depth = 0, float ao = 1, float2 flow = { 0,0 })
{
	output[0][idx] = mask;
	output[1][idx] = normal.x;
	output[2][idx] = normal.y;
	output[3][idx] = normal.z;
	output[4][idx] = depth;
	output[5][idx] = ao;
	output[6][idx] = flow.x;
	output[7][idx] = flow.y;
}

__device__ inline void writeOutputDvr(
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t>& output, int x, int y,
	const float3& rgb = { 0,0,0 }, float alpha = 0, 
	const float3& normal = {0,0,0}, float depth = 0, const float2& flow = {0,0})
{
	output[0][y][x] = rgb.x;
	output[1][y][x] = rgb.y;
	output[2][y][x] = rgb.z;
	output[3][y][x] = alpha;
	output[4][y][x] = normal.x;
	output[5][y][x] = normal.y;
	output[6][y][x] = normal.z;
	output[7][y][x] = depth;
	output[8][y][x] = flow.x;
	output[9][y][x] = flow.y;
}

__device__ inline void writeOutputSamplesDvr(
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t>& output, int idx,
	const float3& rgb = { 0,0,0 }, float alpha = 0,
	const float3& normal = { 0,0,0 }, float depth = 0, const float2& flow = { 0,0 })
{
	output[0][idx] = rgb.x;
	output[1][idx] = rgb.y;
	output[2][idx] = rgb.z;
	output[3][idx] = alpha;
	output[4][idx] = normal.x;
	output[5][idx] = normal.y;
	output[6][idx] = normal.z;
	output[7][idx] = depth;
	output[8][idx] = flow.x;
	output[9][idx] = flow.y;
}

__device__ float customTex3D(cudaTextureObject_t tex, float x, float y, float z,
	std::integral_constant<int, RendererArgs::VolumeFilterMode::TRILINEAR>)
{
	return tex3D<float>(tex, x, y, z);
}

//Source: https://github.com/DannyRuijters/CubicInterpolationCUDA
// Inline calculation of the bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
{
	const T one_frac = 1.0f - fraction;
	const T squared = fraction * fraction;
	const T one_sqd = one_frac * one_frac;

	w0 = 1.0f / 6.0f * one_sqd * one_frac;
	w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
	w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
	w3 = 1.0f / 6.0f * squared * fraction;
}
//Source: https://github.com/DannyRuijters/CubicInterpolationCUDA
//TODO: change to texture object API to support char, short and float textures
__device__ float customTex3D(cudaTextureObject_t tex, float x, float y, float z,
	std::integral_constant<int, RendererArgs::VolumeFilterMode::TRICUBIC>)
{
	const float3 coord = make_float3(x, y, z);
	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	const float3 index = floorf(coord_grid);
	const float3 fraction = coord_grid - index;
	float3 w0, w1, w2, w3;
	bspline_weights(fraction, w0, w1, w2, w3);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;
	const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	typedef float floatN; //return type
	floatN tex000 = tex3D<float>(tex, h0.x, h0.y, h0.z);
	floatN tex100 = tex3D<float>(tex, h1.x, h0.y, h0.z);
	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
	floatN tex010 = tex3D<float>(tex, h0.x, h1.y, h0.z);
	floatN tex110 = tex3D<float>(tex, h1.x, h1.y, h0.z);
	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
	floatN tex001 = tex3D<float>(tex, h0.x, h0.y, h1.z);
	floatN tex101 = tex3D<float>(tex, h1.x, h0.y, h1.z);
	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
	floatN tex011 = tex3D<float>(tex, h0.x, h1.y, h1.z);
	floatN tex111 = tex3D<float>(tex, h1.x, h1.y, h1.z);
	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

	return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
}

__device__ void intersectionRayAABB(
	const float3& rayStart, const float3& rayDir,
	const float3& boxMin, const float3& boxSize,
	float& tmin, float& tmax)
{
	float3 invRayDir = 1.0f / rayDir;
	float t1 = (boxMin.x - rayStart.x) * invRayDir.x;
	float t2 = (boxMin.x + boxSize.x - rayStart.x) * invRayDir.x;
	float t3 = (boxMin.y - rayStart.y) * invRayDir.y;
	float t4 = (boxMin.y + boxSize.y - rayStart.y) * invRayDir.y;
	float t5 = (boxMin.z - rayStart.z) * invRayDir.z;
	float t6 = (boxMin.z + boxSize.z - rayStart.z) * invRayDir.z;
	tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
}

template<int FilterMode>
__device__ float computeAmbientOcclusion(
	cudaTextureObject_t volume_tex, float3 pos, float3 normal, 
	RendererDeviceSettings settings,
	int x, int y)
{
	if (settings.aoSamples == 0) return 1;
	float ao = 0.0;
	//get random rotation vector
	int x2 = x % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
	int y2 = y % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
	float3 noise = make_float3(aoRandomRotations[x2 + AMBIENT_OCCLUSION_RANDOM_ROTATIONS * y2]);
	//compute transformation
	float3 tangent = normalize(noise - normal * dot(noise, normal));
	float3 bitangent = cross(normal, tangent);
	//sample
	const float bias = settings.isovalue;//customTex3D(volume_tex, pos.x, pos.y, pos.z, std::integral_constant<int, FilterMode>());
	for (int i = 0; i < settings.aoSamples; ++i)
	{
		//get hemisphere sample
		float3 sampleT = normalize(make_float3(aoHemisphere[i]));
		//transform to world space
		float3 sampleW = make_float3(
			dot(make_float3(tangent.x, bitangent.x, normal.x), sampleT),
			dot(make_float3(tangent.y, bitangent.y, normal.y), sampleT),
			dot(make_float3(tangent.z, bitangent.z, normal.z), sampleT)
		);
		//shoot ray
		float tmin, tmax;
		intersectionRayAABB(pos, sampleW, settings.boxMin, settings.boxSize, tmin, tmax);
		assert(tmax > 0 && tmin < tmax);
		tmax = min(tmax, settings.aoRadius);
		float value = 1.0;
		for (float sampleDepth = settings.stepsize; sampleDepth <= tmax; sampleDepth += settings.stepsize)
		{
			float3 npos = pos + sampleDepth * sampleW;
			float3 volPos = (npos - settings.boxMin) / settings.boxSize * settings.volumeSize;
			float nval = customTex3D(volume_tex, volPos.x, volPos.y, volPos.z, std::integral_constant<int, FilterMode>());
			if (nval > bias)
			{
				value = .0f;//smoothstep(1, 0, settings.aoRadius / sampleDepth);
				break;
			}
		}
		ao += value;
	}
	return ao / settings.aoSamples;
}

struct RaytraceIsoOutput
{
	float3 posWorld;
	float3 normalWorld;
	float ao;
};
struct RaytraceDvrOutput
{
	float3 color;
	float alpha;
	float3 normalWorld;
	float depth;
};

/**
 * The raytracing kernel for isosurfaces.
 *
 * Input:
 *  - screenPos: integer screen position, needed for AO sampling
 *  - settings: additional render settings
 *  - volume_tex: the 3D volume texture
 *
 * Input: rayStart, rayDir, tmin, tmax.
 * The ray enters the volume at rayStart + tmin*rayDir
 * and leaves the volume at rayStart + tmax*rayDir.
 *
 * Output:
 * Return 'true' if an intersection with the isosurface was found.
 * Then fill 'out' with the position, normal and AO at that position.
 * Else, return 'false'.
 */
template<int FilterMode>
__device__ __inline__ bool RaytraceKernel_Isosurface(
	int2 screenPos, 
	RendererDeviceSettings settings, cudaTextureObject_t volume_tex,
	const float3& rayStart, const float3& rayDir, float tmin, float tmax,
	RaytraceIsoOutput& out)
{
	bool found = false;
	float3 pos = make_float3(0, 0, 0);
	float3 normal = make_float3(0, 0, 0);
	for (float sampleDepth = max(0.0f, tmin); sampleDepth < tmax && !found; sampleDepth += settings.stepsize)
	{
		float3 npos = settings.eyePos + sampleDepth * rayDir;
		float3 volPos = (npos - settings.boxMin) / settings.boxSize * settings.volumeSize;
		float nval = customTex3D(volume_tex, volPos.x, volPos.y, volPos.z, std::integral_constant<int, FilterMode>());
		if (nval > settings.isovalue)
		{
			found = true;
			//TODO: binary search
			//set position to the previous position for AO (slightly outside)
			pos = settings.eyePos + (sampleDepth - settings.stepsize) * rayDir;
			normal.x = 0.5 * (customTex3D(volume_tex, volPos.x + settings.normalStepSize, volPos.y, volPos.z, std::integral_constant<int, FilterMode>())
				- customTex3D(volume_tex, volPos.x - settings.normalStepSize, volPos.y, volPos.z, std::integral_constant<int, FilterMode>()));
			normal.y = 0.5 * (customTex3D(volume_tex, volPos.x, volPos.y + settings.normalStepSize, volPos.z, std::integral_constant<int, FilterMode>())
				- customTex3D(volume_tex, volPos.x, volPos.y - settings.normalStepSize, volPos.z, std::integral_constant<int, FilterMode>()));
			normal.z = 0.5 * (customTex3D(volume_tex, volPos.x, volPos.y, volPos.z + settings.normalStepSize, std::integral_constant<int, FilterMode>())
				- customTex3D(volume_tex, volPos.x, volPos.y, volPos.z - settings.normalStepSize, std::integral_constant<int, FilterMode>()));
			normal = -normal;
		}
	}
	if (found)
	{
		normal = safeNormalize(normal);
		out.posWorld = pos;
		out.normalWorld = normal;

		out.ao = computeAmbientOcclusion<FilterMode>(
			volume_tex, pos - settings.aoBias * rayDir, normal, settings, screenPos.x, screenPos.y);
	}
	return found;
}


/**
 * The raytracing kernel for DVR (Direct Volume Rendering).
 *
 * Input:
 *  - settings: additional render settings
 *  - volumeTex: the 3D volume texture
 *  - tfTexture: transfer function texture
 *
 * Input: rayStart, rayDir, tmin, tmax.
 * The ray enters the volume at rayStart + tmin*rayDir
 * and leaves the volume at rayStart + tmax*rayDir.
 *
 * Output:
 * Returns RGBA value accumulated through the ray direction.
 */
template<int FilterMode>
__device__ __inline__ RaytraceDvrOutput RaytraceKernel_Dvr(
	const RendererDeviceSettings& settings, cudaTextureObject_t volumeTex, cudaTextureObject_t tfTexture,
	const float3& rayStart, const float3& rayDir, float tmin, float tmax)
{
	auto rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
	auto oBuffer = 0.0f;
	auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
	auto depthBuffer = 0.0f;
	for (float sampleDepth = max(0.0f, tmin); sampleDepth < tmax && oBuffer < 0.999f; sampleDepth += settings.stepsize)
	{
		float3 npos = settings.eyePos + sampleDepth * rayDir;
		float3 volPos = (npos - settings.boxMin) / settings.boxSize * settings.volumeSize;
		float nval = customTex3D(volumeTex, volPos.x, volPos.y, volPos.z, std::integral_constant<int, FilterMode>());

		if (nval >= settings.minDensity && nval <= settings.maxDensity)
		{
			nval = (nval - settings.minDensity) / (settings.maxDensity - settings.minDensity);
		}
		else
		{
			continue;
		}

		auto rgba = tex1D<float4>(tfTexture, nval);
		auto opacity = rgba.w * settings.opacityScaling * settings.stepsize;
		opacity = min(1.0, opacity);

		if (opacity > 1e-4)
		{
			//compute normal
			float3 normal;
			normal.x = 0.5 * (customTex3D(volumeTex, volPos.x + settings.normalStepSize, volPos.y, volPos.z, std::integral_constant<int, FilterMode>())
                - customTex3D(volumeTex, volPos.x - settings.normalStepSize, volPos.y, volPos.z, std::integral_constant<int, FilterMode>()));
			normal.y = 0.5 * (customTex3D(volumeTex, volPos.x, volPos.y + settings.normalStepSize, volPos.z, std::integral_constant<int, FilterMode>())
                - customTex3D(volumeTex, volPos.x, volPos.y - settings.normalStepSize, volPos.z, std::integral_constant<int, FilterMode>()));
			normal.z = 0.5 * (customTex3D(volumeTex, volPos.x, volPos.y, volPos.z + settings.normalStepSize, std::integral_constant<int, FilterMode>())
                - customTex3D(volumeTex, volPos.x, volPos.y, volPos.z - settings.normalStepSize, std::integral_constant<int, FilterMode>()));
			normal = safeNormalize(normal);

			if (settings.useShading)
			{
				//perform phong shading
				float3 color = make_float3(0);
				float3 col = make_float3(rgba);
				color += settings.shading.ambientLightColor * col; //ambient light
				color += settings.shading.diffuseLightColor * col *
                    abs(dot(normal, settings.shading.lightDirection)); //diffuse
				float3 reflect = 2 * dot(settings.shading.lightDirection, normal) *
                    normal - settings.shading.lightDirection;
				color += settings.shading.specularLightColor * (
                    ((settings.shading.specularExponent + 2) / (2 * M_PI)) *
                    pow(clamp(dot(reflect, rayDir), 0.0f, 1.0f), settings.shading.specularExponent));
				//set as final color, keep alpha
				rgba = make_float4(color, rgba.w);
			}

			rgbBuffer += (1.0f - oBuffer) * opacity * make_float3(rgba.x, rgba.y, rgba.z);
			normalBuffer += (1.0f - oBuffer) * opacity * normal;
			depthBuffer += (1.0f - oBuffer) * opacity * sampleDepth;
			oBuffer += (1.0f - oBuffer) * opacity;
		}
	}

	return { rgbBuffer, oBuffer, normalBuffer, depthBuffer};
}

/**
 * The rendering kernel with the parallel loop over the pixels and output handling
 */
template<int FilterMode>
__global__ void IsosurfaceKernel(dim3 virtual_size, 
	RendererDeviceSettings settings, cudaTextureObject_t volume_tex,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_2D_LOOP(x_, y_, virtual_size)
	
	int x = x_ + settings.viewport.x;
	int y = y_ + settings.viewport.y;
	float posx = ((x+0.5f) / settings.screenSize.x) * 2 - 1;
	float posy = ((y+0.5f) / settings.screenSize.y) * 2 - 1;

	//target world position
	float4 screenPos = make_float4(posx, posy, 0.9, 1);
	float4 worldPos = matmul(settings.currentViewMatrixInverse, screenPos);
	worldPos /= worldPos.w;

	//ray direction
	float3 rayDir = normalize(make_float3(worldPos) - settings.eyePos);

	//entry, exit points
	float tmin, tmax;
	intersectionRayAABB(settings.eyePos, rayDir, settings.boxMin, settings.boxSize, tmin, tmax);
	if (tmax < 0 || tmin > tmax)
	{
		writeOutputIso(output, x_, y_);
		continue;
	}

	//perform stepping
	RaytraceIsoOutput out;
	bool found = RaytraceKernel_Isosurface<FilterMode>(
		make_int2(x, y), settings, volume_tex,
		settings.eyePos, rayDir, tmin, tmax, out);

	if (found)
	{
		float4 screenCurrent = matmul(settings.currentViewMatrix, make_float4(out.posWorld, 1.0));
		screenCurrent /= screenCurrent.w;
		float4 screenNext = matmul(settings.nextViewMatrix, make_float4(out.posWorld, 1.0));
		screenNext /= screenNext.w;
		//evaluate depth and flow
		float mask = 1;
		float depth = screenCurrent.z;
		float2 flow = 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y);
		float3 normalScreen = make_float3(matmul(settings.normalMatrix, make_float4(-out.normalWorld, 0)));
		//write output
		writeOutputIso(output, x_, y_, mask, normalScreen, depth, out.ao, flow);
	} else
	{
		writeOutputIso(output, x_, y_);
	}

	CUMAT_KERNEL_2D_LOOP_END
}

/**
 * The rendering kernel with the parallel loop over the pixels and output handling
 */
template<int FilterMode>
__global__ void DvrKernel(dim3 virtual_size,
	RendererDeviceSettings settings, cudaTextureObject_t volumeTex, cudaTextureObject_t tfTexture,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_2D_LOOP(x_, y_, virtual_size)

	int x = x_ + settings.viewport.x;
	int y = y_ + settings.viewport.y;
	float posx = ((x + 0.5f) / settings.screenSize.x) * 2 - 1;
	float posy = ((y + 0.5f) / settings.screenSize.y) * 2 - 1;

	//target world position
	float4 screenPos = make_float4(posx, posy, 0.9, 1);
	float4 worldPos = matmul(settings.currentViewMatrixInverse, screenPos);
	worldPos /= worldPos.w;

	//ray direction
	float3 rayDir = normalize(make_float3(worldPos) - settings.eyePos);

	//entry, exit points
	float tmin, tmax;
	intersectionRayAABB(settings.eyePos, rayDir, settings.boxMin, settings.boxSize, tmin, tmax);
	if (tmax < 0 || tmin > tmax)
	{
		writeOutputDvr(output, x_, y_);
		continue;
	}

	const auto out = RaytraceKernel_Dvr<FilterMode>(
		settings, volumeTex, tfTexture,
		settings.eyePos, rayDir, tmin, tmax);

	//evaluate flow and depth
	float depth = out.alpha > 1e-5 ? out.depth / out.alpha : 0;
	float3 posWorld = settings.eyePos + (depth - settings.stepsize) * rayDir;
	float4 screenCurrent = matmul(settings.currentViewMatrix, make_float4(posWorld, 1.0));
	screenCurrent /= screenCurrent.w;
	float4 screenNext = matmul(settings.nextViewMatrix, make_float4(posWorld, 1.0));
	screenNext /= screenNext.w;
	float2 flow = out.alpha > 1e-5
		? 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y)
		: make_float2(0, 0);
	float depthScreen = out.alpha > 1e-5 ? screenCurrent.z : 0.0f;
	//evaluate normal
	float3 normalWorld = safeNormalize(out.normalWorld);
	float3 normalScreen = make_float3(matmul(settings.normalMatrix, make_float4(-normalWorld, 0)));

	writeOutputDvr(output, x_, y_,
		out.color, out.alpha, normalScreen, depthScreen, flow);

	CUMAT_KERNEL_2D_LOOP_END
}

/**
 * The rendering kernel with the parallel loop over the pixels and output handling
 */
template<int FilterMode>
__global__ void IsosurfaceSamplesKernel(dim3 virtual_size,
	RendererDeviceSettings settings, cudaTextureObject_t volume_tex,
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> input,
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_1D_LOOP(idx, virtual_size)

	float x = input[0][idx];
	float y = input[1][idx];
	if (x < settings.viewport.x || y <= settings.viewport.y ||
		x > settings.viewport.z || y >= settings.viewport.w)
	{
		writeOutputSamplesIso(output, idx);
		continue;
	}
	float posx = ((x + 0.5f) / settings.screenSize.x) * 2 - 1;
	float posy = ((y + 0.5f) / settings.screenSize.y) * 2 - 1;

	//target world position
	float4 screenPos = make_float4(posx, posy, 0.9, 1);
	float4 worldPos = matmul(settings.currentViewMatrixInverse, screenPos);
	worldPos /= worldPos.w;

	//ray direction
	float3 rayDir = normalize(make_float3(worldPos) - settings.eyePos);

	//entry, exit points
	float tmin, tmax;
	intersectionRayAABB(settings.eyePos, rayDir, settings.boxMin, settings.boxSize, tmin, tmax);
	if (tmax < 0 || tmin > tmax)
	{
		writeOutputSamplesIso(output, idx);
		continue;
	}

	//perform stepping
	RaytraceIsoOutput out;
	bool found = RaytraceKernel_Isosurface<FilterMode>(
		make_int2(x, y), settings, volume_tex,
		settings.eyePos, rayDir, tmin, tmax, out);

	if (found)
	{
		float4 screenCurrent = matmul(settings.currentViewMatrix, make_float4(out.posWorld, 1.0));
		screenCurrent /= screenCurrent.w;
		float4 screenNext = matmul(settings.nextViewMatrix, make_float4(out.posWorld, 1.0));
		screenNext /= screenNext.w;
		//evaluate depth and flow
		float mask = 1;
		float depth = screenCurrent.z;
		float2 flow = 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y);
		float3 normalScreen = make_float3(matmul(settings.normalMatrix, make_float4(-out.normalWorld, 0)));
		//write output
		writeOutputSamplesIso(output, idx, mask, normalScreen, depth, out.ao, flow);
	}
	else
	{
		writeOutputSamplesIso(output, idx);
	}

	CUMAT_KERNEL_1D_LOOP_END
}

/**
 * The rendering kernel with the parallel loop over the pixels and output handling
 */
template<int FilterMode>
__global__ void DVRSamplesKernel(dim3 virtual_size,
	RendererDeviceSettings settings, cudaTextureObject_t volume_tex, cudaTextureObject_t tfTexture,
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> input,
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_1D_LOOP(idx, virtual_size)
	{
		float x = input[0][idx];
		float y = input[1][idx];
		if (x < settings.viewport.x || y <= settings.viewport.y ||
            x > settings.viewport.z || y >= settings.viewport.w)
		{
			writeOutputSamplesDvr(output, idx);
			continue;
		}
		float posx = ((x + 0.5f) / settings.screenSize.x) * 2 - 1;
		float posy = ((y + 0.5f) / settings.screenSize.y) * 2 - 1;

		//target world position
		float4 screenPos = make_float4(posx, posy, 0.9, 1);
		float4 worldPos = matmul(settings.currentViewMatrixInverse, screenPos);
		worldPos /= worldPos.w;

		//ray direction
		float3 rayDir = normalize(make_float3(worldPos) - settings.eyePos);

		//entry, exit points
		float tmin, tmax;
		intersectionRayAABB(settings.eyePos, rayDir, settings.boxMin, settings.boxSize, tmin, tmax);
		if (tmax < 0 || tmin > tmax)
		{
			writeOutputSamplesDvr(output, idx);
			continue;
		}

		//perform stepping
		RaytraceDvrOutput out = RaytraceKernel_Dvr<FilterMode>(
            settings, volume_tex, tfTexture,
            settings.eyePos, rayDir, tmin, tmax);

		//evaluate flow and depth
		float depth = out.alpha > 1e-5 ? out.depth / out.alpha : 0;
		float3 posWorld = settings.eyePos + (depth - settings.stepsize) * rayDir;
		float4 screenCurrent = matmul(settings.currentViewMatrix, make_float4(posWorld, 1.0));
		screenCurrent /= screenCurrent.w;
		float4 screenNext = matmul(settings.nextViewMatrix, make_float4(posWorld, 1.0));
		screenNext /= screenNext.w;
		float2 flow = out.alpha > 1e-5
			? 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y)
			: make_float2(0, 0);
		float depthScreen = out.alpha > 1e-5 ? screenCurrent.z : 0.0f;
		//evaluate normal
		float3 normalWorld = safeNormalize(out.normalWorld);
		float3 normalScreen = make_float3(matmul(settings.normalMatrix, make_float4(-normalWorld, 0)));

		writeOutputSamplesDvr(output, idx,
			out.color, out.alpha, normalScreen, depthScreen, flow);
	}
	CUMAT_KERNEL_1D_LOOP_END
}

/**
 * The rendering kernel with the parallel loop over the pixels and output handling
 */
template<int FilterMode>
__global__ void IsosurfaceStepsizeKernel(dim3 virtual_size,
	RendererDeviceSettings settings, cudaTextureObject_t volume_tex,
	const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> stepsize,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_2D_LOOP(x_, y_, virtual_size)

	int x = x_ + settings.viewport.x;
	int y = y_ + settings.viewport.y;
	float posx = ((x + 0.5f) / settings.screenSize.x) * 2 - 1;
	float posy = ((y + 0.5f) / settings.screenSize.y) * 2 - 1;

	//target world position
	float4 screenPos = make_float4(posx, posy, 0.9, 1);
	float4 worldPos = matmul(settings.currentViewMatrixInverse, screenPos);
	worldPos /= worldPos.w;

	//ray direction
	float3 rayDir = normalize(make_float3(worldPos) - settings.eyePos);

	//entry, exit points
	float tmin, tmax;
	intersectionRayAABB(settings.eyePos, rayDir, settings.boxMin, settings.boxSize, tmin, tmax);
	if (tmax < 0 || tmin > tmax)
	{
		writeOutputIso(output, x, y);
		continue;
	}

	//perform stepping
	RendererDeviceSettings settings2 = settings;
	settings2.stepsize *= max(0.0f, stepsize[y][x]);
	RaytraceIsoOutput out;
	bool found = RaytraceKernel_Isosurface<FilterMode>(
		make_int2(x, y), settings2, volume_tex,
		settings.eyePos, rayDir, tmin, tmax, out);

	if (found)
	{
		float4 screenCurrent = matmul(settings.currentViewMatrix, make_float4(out.posWorld, 1.0));
		screenCurrent /= screenCurrent.w;
		float4 screenNext = matmul(settings.nextViewMatrix, make_float4(out.posWorld, 1.0));
		screenNext /= screenNext.w;
		//evaluate depth and flow
		float mask = 1;
		float depth = screenCurrent.z;
		float2 flow = 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y);
		float3 normalScreen = make_float3(matmul(settings.normalMatrix, make_float4(-out.normalWorld, 0)));
		//write output
		writeOutputIso(output, x, y, mask, normalScreen, depth, out.ao, flow);
	}
	else
	{
		writeOutputIso(output, x, y);
	}

	CUMAT_KERNEL_2D_LOOP_END
}

/**
 * The rendering kernel with the parallel loop over the pixels and output handling
 */
template<int FilterMode>
__global__ void DvrStepsizeKernel(dim3 virtual_size,
	RendererDeviceSettings settings, cudaTextureObject_t volumeTex, cudaTextureObject_t tfTexture,
	const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> stepsize,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_2D_LOOP(x_, y_, virtual_size)

	int x = x_ + settings.viewport.x;
	int y = y_ + settings.viewport.y;
	float posx = ((x + 0.5f) / settings.screenSize.x) * 2 - 1;
	float posy = ((y + 0.5f) / settings.screenSize.y) * 2 - 1;

	//target world position
	float4 screenPos = make_float4(posx, posy, 0.9, 1);
	float4 worldPos = matmul(settings.currentViewMatrixInverse, screenPos);
	worldPos /= worldPos.w;

	//ray direction
	float3 rayDir = normalize(make_float3(worldPos) - settings.eyePos);

	//entry, exit points
	float tmin, tmax;
	intersectionRayAABB(settings.eyePos, rayDir, settings.boxMin, settings.boxSize, tmin, tmax);
	if (tmax < 0 || tmin > tmax)
	{
		writeOutputDvr(output, x, y);
		continue;
	}

	RendererDeviceSettings settings2 = settings;
	settings2.stepsize *= max(0.0f, stepsize[y][x]);
	const auto out = RaytraceKernel_Dvr<FilterMode>(
		settings2, volumeTex, tfTexture,
		settings.eyePos, rayDir, tmin, tmax);

	//evaluate flow and depth
	float depth = out.alpha > 1e-5 ? out.depth / out.alpha : 0;
	float3 posWorld = settings.eyePos + (depth - settings.stepsize) * rayDir;
	float4 screenCurrent = matmul(settings.currentViewMatrix, make_float4(posWorld, 1.0));
	screenCurrent /= screenCurrent.w;
	float4 screenNext = matmul(settings.nextViewMatrix, make_float4(posWorld, 1.0));
	screenNext /= screenNext.w;
	float2 flow = out.alpha > 1e-5
		? 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y)
		: make_float2(0, 0);
	float depthScreen = out.alpha > 1e-5 ? screenCurrent.z : 0.0f;
	//evaluate normal
	float3 normalWorld = safeNormalize(out.normalWorld);
	float3 normalScreen = make_float3(matmul(settings.normalMatrix, make_float4(-normalWorld, 0)));

	writeOutputDvr(output, x, y,
		out.color, out.alpha, normalScreen, depthScreen, flow);

	CUMAT_KERNEL_2D_LOOP_END
}

template<int N>
__global__ void ScatterSamplesKernel(dim3 virtual_size,
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> sample_positions,
	torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> samples,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> image_out,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> sample_mask_out)
{
	CUMAT_KERNEL_1D_LOOP(idx, virtual_size)

	int x = round(sample_positions[0][idx]);
	int y = round(sample_positions[1][idx]);
	if (x < 0 || y < 0 || x >= image_out.size(2) || y >= image_out.size(1))
		continue;

#pragma unroll
	for (int i = 0; i < N; ++i)
		image_out[i][y][x] = samples[i][idx];
	sample_mask_out[0][y][x] = 1.0f;

	CUMAT_KERNEL_1D_LOOP_END
}

namespace
{
	template<int N>
	struct Defaults
	{
		float values[N];
	};
}
template<int N>
__global__ void FillDefaultKernel(dim3 virtual_size,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> image_out,
	Defaults<N> defaults)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
#pragma unroll
		for (int i = 0; i < N; ++i)
			image_out[i][y][x] = defaults.values[i];
	CUMAT_KERNEL_2D_LOOP_END
}

//=========================================
// RENDERER_LAUNCHER
//=========================================

void render_gpu(const Volume* volume, const RendererArgs* args, torch::Tensor& output, cudaStream_t stream)
{
	CHECK_INPUT(output);
	const Volume::MipmapLevel* data = volume->getLevel(args->mipmapLevel);
	TORCH_CHECK(data != nullptr, "mipmap level must exist");
	
	//set settings
	RendererDeviceSettings s;
	s.screenSize = make_float2(args->cameraResolutionX, args->cameraResolutionY);
	s.volumeSize = make_float3(data->sizeX(), data->sizeY(), data->sizeZ());
	s.binarySearchSteps = args->binarySearchSteps;
	s.stepsize = args->stepsize / std::max({data->sizeX(), data->sizeY(), data->sizeZ()});
	s.normalStepSize = 0.5f;
	s.boxSize = make_float3(
		volume->worldSizeX(),
		volume->worldSizeY(),
		volume->worldSizeZ());
	s.boxMin = make_float3(-s.boxSize.x / 2, -s.boxSize.y / 2, -s.boxSize.z / 2);
	s.isovalue = args->isovalue;
	s.aoBias = args->aoBias;
	s.aoRadius = args->aoRadius;
	s.aoSamples = args->aoSamples;
	s.eyePos = args->cameraOrigin;
	s.viewport = args->cameraViewport;
	if (s.viewport.z < 0) s.viewport.z = args->cameraResolutionX;
	if (s.viewport.w < 0) s.viewport.w = args->cameraResolutionY;
	Camera::computeMatrices(
		args->cameraOrigin, args->cameraLookAt, args->cameraUp,
		args->cameraFovDegrees, args->cameraResolutionX, args->cameraResolutionY, args->nearClip, args->farClip,
		s.currentViewMatrix, s.currentViewMatrixInverse, s.normalMatrix);
	static float4 lastViewMatrix[4] = {
		make_float4(1,0,0,0), make_float4(0,1,0,0),
		make_float4(0,0,1,0), make_float4(0,0,0,1)};
	memcpy(s.nextViewMatrix, lastViewMatrix, sizeof(float4) * 4);
	memcpy(lastViewMatrix, s.currentViewMatrix, sizeof(float4) * 4);
	s.opacityScaling = args->opacityScaling;
	s.minDensity = args->minDensity;
	s.maxDensity = args->maxDensity;
	s.useShading = args->dvrUseShading;
	s.shading = args->shading;

	//launch kernel
	cuMat::Context& ctx = cuMat::Context::current();

	switch (args->renderMode)
	{
	case renderer::RendererArgs::ISO_UNSHADED:
		if (args->volumeFilterMode == RendererArgs::TRILINEAR) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, IsosurfaceKernel<RendererArgs::TRILINEAR>);
			IsosurfaceKernel<RendererArgs::TRILINEAR>
				<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size, s, data->dataTexGpu(), output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		else if (args->volumeFilterMode == RendererArgs::TRICUBIC) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, IsosurfaceKernel<RendererArgs::TRICUBIC>);
			IsosurfaceKernel<RendererArgs::TRICUBIC>
				<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size, s, data->dataTexGpu(), output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		break;

	case renderer::RendererArgs::DVR:
		static TfTexture1D rendererTfTexture;
		rendererTfTexture.updateIfChanged(args->densityAxisOpacity, args->opacityAxis, args->densityAxisColor, args->colorAxis);
		
		if (args->volumeFilterMode == RendererArgs::TRILINEAR) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, DvrKernel<RendererArgs::TRILINEAR>);
			DvrKernel<RendererArgs::TRILINEAR>
				<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size, s, data->dataTexGpu(), rendererTfTexture.getTextureObject(), output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		else if (args->volumeFilterMode == RendererArgs::TRICUBIC) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, DvrKernel<RendererArgs::TRICUBIC>);
			DvrKernel<RendererArgs::TRICUBIC>
				<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size, s, data->dataTexGpu(), rendererTfTexture.getTextureObject(), output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		break;
	}
	CUMAT_CHECK_ERROR();

	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

void render_samples_gpu(const Volume* volume, const RendererArgs* args, const torch::Tensor& sample_positions,
	torch::Tensor& samples_out, cudaStream_t stream)
{
	//TODO: support args->renderMode == DVR
	
	CHECK_INPUT(sample_positions);
	CHECK_INPUT(samples_out);
	AT_ASSERTM(sample_positions.dim() == 2, "sample_positions has to be 2D");
	AT_ASSERTM(samples_out.dim() == 2, "samples_out has to be 2D");
	AT_ASSERTM(sample_positions.size(0) == 2, "sample_positions has to be of shape 2*N");
	size_t N = sample_positions.size(1);
	AT_ASSERTM(samples_out.size(1) == N, "sample_positions and samples_out don't agree in the number of samples");
	
	const Volume::MipmapLevel* data = volume->getLevel(args->mipmapLevel);
	TORCH_CHECK(data != nullptr, "mipmap level must exist");

	//set settings
	RendererDeviceSettings s;
	s.screenSize = make_float2(args->cameraResolutionX, args->cameraResolutionY);
	s.volumeSize = make_float3(data->sizeX(), data->sizeY(), data->sizeZ());
	s.binarySearchSteps = args->binarySearchSteps;
	s.stepsize = args->stepsize / std::max({ data->sizeX(), data->sizeY(), data->sizeZ() });
	s.normalStepSize = 0.5f;
	s.boxSize = make_float3(
		volume->worldSizeX(),
		volume->worldSizeY(),
		volume->worldSizeZ());
	s.boxMin = make_float3(-s.boxSize.x / 2, -s.boxSize.y / 2, -s.boxSize.z / 2);
	s.isovalue = args->isovalue;
	s.aoBias = args->aoBias;
	s.aoRadius = args->aoRadius;
	s.aoSamples = args->aoSamples;
	s.eyePos = args->cameraOrigin;
	s.viewport = args->cameraViewport;
	if (s.viewport.z < 0) s.viewport.z = args->cameraResolutionX;
	if (s.viewport.w < 0) s.viewport.w = args->cameraResolutionY;
	Camera::computeMatrices(
		args->cameraOrigin, args->cameraLookAt, args->cameraUp,
		args->cameraFovDegrees, args->cameraResolutionX, args->cameraResolutionY, args->nearClip, args->farClip,
		s.currentViewMatrix, s.currentViewMatrixInverse, s.normalMatrix);
	static float4 lastViewMatrix[4] = {
		make_float4(1,0,0,0), make_float4(0,1,0,0),
		make_float4(0,0,1,0), make_float4(0,0,0,1) };
	memcpy(s.nextViewMatrix, lastViewMatrix, sizeof(float4) * 4);
	memcpy(lastViewMatrix, s.currentViewMatrix, sizeof(float4) * 4);
	s.opacityScaling = args->opacityScaling;
	s.minDensity = args->minDensity;
	s.maxDensity = args->maxDensity;
	s.useShading = args->dvrUseShading;
	s.shading = args->shading;

	//launch kernel
	cuMat::Context& ctx = cuMat::Context::current();
	switch (args->renderMode)
	{
	case renderer::RendererArgs::ISO_UNSHADED:
		AT_ASSERTM(samples_out.size(0) == IsoRendererOutputChannels, "samples_out has to be of shape 8*N");
        if (args->volumeFilterMode == RendererArgs::TRILINEAR) {
        	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(N, IsosurfaceSamplesKernel<RendererArgs::TRILINEAR>);
        	IsosurfaceSamplesKernel<RendererArgs::TRILINEAR>
                <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
                (cfg.virtual_size, s, data->dataTexGpu(),
                    sample_positions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                    samples_out.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());
        }
        else if (args->volumeFilterMode == RendererArgs::TRICUBIC) {
        	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(N, IsosurfaceSamplesKernel<RendererArgs::TRICUBIC>);
        	IsosurfaceSamplesKernel<RendererArgs::TRICUBIC>
                <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
                (cfg.virtual_size, s, data->dataTexGpu(),
                    sample_positions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                    samples_out.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());
        }
		break;
	case renderer::RendererArgs::DVR:
		AT_ASSERTM(samples_out.size(0) == DvrRendererOutputChannels, "samples_out has to be of shape 10*N");
		static TfTexture1D rendererTfTexture;
		rendererTfTexture.updateIfChanged(args->densityAxisOpacity, args->opacityAxis, args->densityAxisColor, args->colorAxis);
		if (args->volumeFilterMode == RendererArgs::TRILINEAR) {
        	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(N, IsosurfaceSamplesKernel<RendererArgs::TRILINEAR>);
			DVRSamplesKernel<RendererArgs::TRILINEAR>
                <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
                (cfg.virtual_size, s, data->dataTexGpu(), rendererTfTexture.getTextureObject(),
                    sample_positions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                    samples_out.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());
        }
        else if (args->volumeFilterMode == RendererArgs::TRICUBIC) {
        	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(N, IsosurfaceSamplesKernel<RendererArgs::TRICUBIC>);
			DVRSamplesKernel<RendererArgs::TRICUBIC>
                <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
                (cfg.virtual_size, s, data->dataTexGpu(), rendererTfTexture.getTextureObject(),
                    sample_positions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                    samples_out.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());
        }
        break;
	}
	CUMAT_CHECK_ERROR();

	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

void render_adaptive_stepsize_gpu(const Volume* volume, const RendererArgs* args, const torch::Tensor& stepsize,
	torch::Tensor& output, cudaStream_t stream)
{
	CHECK_INPUT(output);
	CHECK_INPUT(stepsize);
	TORCH_CHECK(output.dim() == 3, "Output must be of shape (BxHxW), but is not 3-dimensional");
	TORCH_CHECK(stepsize.dim() == 2, "Stepsize must be of shape (HxW), but is not 2-dimensional");
	TORCH_CHECK(output.size(1) == stepsize.size(0), "Stepsize and Output are not compatible");
	TORCH_CHECK(output.size(2) == stepsize.size(1), "Stepsize and Output are not compatible");
	
	const Volume::MipmapLevel* data = volume->getLevel(args->mipmapLevel);
	TORCH_CHECK(data != nullptr, "mipmap level must exist");
	
	//set settings
	RendererDeviceSettings s;
	s.screenSize = make_float2(args->cameraResolutionX, args->cameraResolutionY);
	s.volumeSize = make_float3(data->sizeX(), data->sizeY(), data->sizeZ());
	s.binarySearchSteps = args->binarySearchSteps;
	//stepsize scaling factor
	s.stepsize = args->stepsize / std::max({data->sizeX(), data->sizeY(), data->sizeZ()});
	s.normalStepSize = 0.5f;
	s.boxSize = make_float3(
		volume->worldSizeX(),
		volume->worldSizeY(),
		volume->worldSizeZ());
	s.boxMin = make_float3(-s.boxSize.x / 2, -s.boxSize.y / 2, -s.boxSize.z / 2);
	s.isovalue = args->isovalue;
	s.aoBias = args->aoBias;
	s.aoRadius = args->aoRadius;
	s.aoSamples = args->aoSamples;
	s.eyePos = args->cameraOrigin;
	s.viewport = args->cameraViewport;
	if (s.viewport.z < 0) s.viewport.z = args->cameraResolutionX;
	if (s.viewport.w < 0) s.viewport.w = args->cameraResolutionY;
	Camera::computeMatrices(
		args->cameraOrigin, args->cameraLookAt, args->cameraUp,
		args->cameraFovDegrees, args->cameraResolutionX, args->cameraResolutionY, args->nearClip, args->farClip,
		s.currentViewMatrix, s.currentViewMatrixInverse, s.normalMatrix);
	static float4 lastViewMatrix[4] = {
		make_float4(1,0,0,0), make_float4(0,1,0,0),
		make_float4(0,0,1,0), make_float4(0,0,0,1)};
	memcpy(s.nextViewMatrix, lastViewMatrix, sizeof(float4) * 4);
	memcpy(lastViewMatrix, s.currentViewMatrix, sizeof(float4) * 4);
	s.opacityScaling = args->opacityScaling;
	s.minDensity = args->minDensity;
	s.maxDensity = args->maxDensity;
	s.useShading = args->dvrUseShading;
	s.shading = args->shading;

	//launch kernel
	cuMat::Context& ctx = cuMat::Context::current();

	switch (args->renderMode)
	{
	case renderer::RendererArgs::ISO_UNSHADED:
		if (args->volumeFilterMode == RendererArgs::TRILINEAR) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, IsosurfaceStepsizeKernel<RendererArgs::TRILINEAR>);
			IsosurfaceStepsizeKernel<RendererArgs::TRILINEAR>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size, s, data->dataTexGpu(),
					stepsize.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
					output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		else if (args->volumeFilterMode == RendererArgs::TRICUBIC) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, IsosurfaceStepsizeKernel<RendererArgs::TRICUBIC>);
			IsosurfaceStepsizeKernel<RendererArgs::TRICUBIC>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size, s, data->dataTexGpu(), 
					stepsize.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
					output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		break;

	case renderer::RendererArgs::DVR:
		static TfTexture1D rendererTfTexture;
		rendererTfTexture.updateIfChanged(args->densityAxisOpacity, args->opacityAxis, args->densityAxisColor, args->colorAxis);
		
		if (args->volumeFilterMode == RendererArgs::TRILINEAR) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, DvrStepsizeKernel<RendererArgs::TRILINEAR>);
			DvrStepsizeKernel<RendererArgs::TRILINEAR>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size, s, data->dataTexGpu(),
					rendererTfTexture.getTextureObject(),
					stepsize.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(), 
					output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		else if (args->volumeFilterMode == RendererArgs::TRICUBIC) {
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y, DvrStepsizeKernel<RendererArgs::TRICUBIC>);
			DvrStepsizeKernel<RendererArgs::TRICUBIC>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size, s, data->dataTexGpu(),
					rendererTfTexture.getTextureObject(),
					stepsize.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(), 
					output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		}
		break;
	}
	CUMAT_CHECK_ERROR();
}

void scatter_samples_to_image_gpu(const torch::Tensor& sample_positions, const torch::Tensor& samples,
	torch::Tensor& image_out, torch::Tensor& sample_mask_out, cudaStream_t stream)
{
	CHECK_INPUT(sample_positions);
	CHECK_INPUT(samples);
	AT_ASSERTM(sample_positions.dim() == 2, "sample_positions has to be 2D");
	AT_ASSERTM(samples.dim() == 2, "samples_out has to be 2D");
	AT_ASSERTM(sample_positions.size(0) == 2, "sample_positions has to be of shape 2*N");
	size_t N = sample_positions.size(1);
	AT_ASSERTM(samples.size(1) == N, "sample_positions and samples_out don't agree in the number of samples");

	CHECK_INPUT(image_out);
	AT_ASSERTM(image_out.dim() == 3, "image_out has to be 3D");
	
	CHECK_INPUT(sample_mask_out);
	AT_ASSERTM(sample_mask_out.dim() == 3, "sample_mask_out must be 3D");
	AT_ASSERTM(sample_mask_out.size(0) == 1, "sample_mask_out must have one channel");
	AT_ASSERTM(image_out.size(1) == sample_mask_out.size(1), "image_out and sample_mask_out must have the same spatial size");
	AT_ASSERTM(image_out.size(2) == sample_mask_out.size(2), "image_out and sample_mask_out must have the same spatial size");

	sample_mask_out.fill_(0.0f);
	
	cuMat::Context& ctx = cuMat::Context::current();
	if (samples.size(0) == renderer::IsoRendererOutputChannels)
	{
		AT_ASSERTM(samples.size(0) == renderer::IsoRendererOutputChannels, "samples_out has to be of shape 8*N");
		AT_ASSERTM(image_out.size(0) == renderer::IsoRendererOutputChannels, "image_out has to be of shape 8*H*W");
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(N, ScatterSamplesKernel<renderer::IsoRendererOutputChannels>);
		ScatterSamplesKernel<renderer::IsoRendererOutputChannels>
            <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
            (cfg.virtual_size,
                sample_positions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                samples.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                image_out.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
				sample_mask_out.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		CUMAT_CHECK_ERROR();
	}
	else if (samples.size(0) == renderer::DvrRendererOutputChannels)
	{
		AT_ASSERTM(samples.size(0) == renderer::DvrRendererOutputChannels, "samples_out has to be of shape 10*N");
		AT_ASSERTM(image_out.size(0) == renderer::DvrRendererOutputChannels, "image_out has to be of shape 10*H*W");
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(N, ScatterSamplesKernel<renderer::DvrRendererOutputChannels>);
		ScatterSamplesKernel<renderer::DvrRendererOutputChannels>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				sample_positions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
				samples.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
				image_out.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
				sample_mask_out.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
		CUMAT_CHECK_ERROR();
	}
	else
	{
		AT_ASSERTM(false, "only IsoRendererOutputChannels or DvrRendererOutputChannels supported as channel count");
	}
}

int64_t initializeRenderer()
{
	auto params = computeAmbientOcclusionParameters(MAX_AMBIENT_OCCLUSION_SAMPLES, AMBIENT_OCCLUSION_RANDOM_ROTATIONS);
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(aoHemisphere, std::get<0>(params).data(), sizeof(float4)*MAX_AMBIENT_OCCLUSION_SAMPLES));
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(aoRandomRotations, std::get<1>(params).data(), sizeof(float4)*AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS));
	return 1;
}

torch::Tensor Render()
{
	//ensure the volume is available on GPU
	TheVolume->getLevel(TheRendererArgs.mipmapLevel)->copyCpuToGpu();

	//create output tensor
	at::TensorOptions opt;
	at::Tensor output;
	
	int output_channels = 0;
	if (TheRendererArgs.renderMode == RendererArgs::ISO_UNSHADED)
	{
		output_channels = IsoRendererOutputChannels;
	}
	else if (TheRendererArgs.renderMode == RendererArgs::DVR)
	{
		output_channels = DvrRendererOutputChannels;
	}

	output = at::zeros({ output_channels, TheRendererArgs.cameraResolutionY, TheRendererArgs.cameraResolutionX },
		at::dtype(at::kFloat).device(at::kCUDA));

	//call renderer
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	render_gpu(TheVolume.get(), &TheRendererArgs, output, at::cuda::getCurrentCUDAStream());
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	
	return output;
}

torch::Tensor RenderSamples(const torch::Tensor& sample_position)
{
	//ensure the volume is available on GPU
	TheVolume->getLevel(0)->copyCpuToGpu();

	int C = TheRendererArgs.renderMode == RendererArgs::ISO_UNSHADED
		? IsoRendererOutputChannels : DvrRendererOutputChannels;
	
	//create output tensor
	int64_t N = sample_position.size(1);
	at::Tensor output = at::empty({ C, N },
		at::dtype(at::kFloat).device(at::kCUDA));

	//call renderer
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	render_samples_gpu(TheVolume.get(), &TheRendererArgs, sample_position, output, at::cuda::getCurrentCUDAStream());
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	
	return output;
}

torch::Tensor RenderAdaptiveStepsize(const torch::Tensor& stepsizes)
{
	//ensure the volume is available on GPU
	TheVolume->getLevel(TheRendererArgs.mipmapLevel)->copyCpuToGpu();

	//create output tensor
	at::TensorOptions opt;
	at::Tensor output;

	int output_channels = 0;
	if (TheRendererArgs.renderMode == RendererArgs::ISO_UNSHADED)
	{
		output_channels = IsoRendererOutputChannels;
	}
	else if (TheRendererArgs.renderMode == RendererArgs::DVR)
	{
		output_channels = DvrRendererOutputChannels;
	}

	output = at::zeros({ output_channels, TheRendererArgs.cameraResolutionY, TheRendererArgs.cameraResolutionX },
		at::dtype(at::kFloat).device(at::kCUDA));

	//call renderer
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	render_adaptive_stepsize_gpu(
		TheVolume.get(), &TheRendererArgs, stepsizes, output, at::cuda::getCurrentCUDAStream());
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	return output;
}

torch::Tensor ScatterSamplesToImage(const torch::Tensor& sample_position, const torch::Tensor& samples, int64_t width,
	int64_t height, std::vector<double> default_values)
{
	const int C = samples.size(0);
	AT_ASSERTM(default_values.size() == C, "number of channels in the samples must match the number of default values");
	
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	at::Tensor output;
	//create default tensor
	if (C == IsoRendererOutputChannels)
	{
		Defaults<IsoRendererOutputChannels> defaults;
		for (int i = 0; i < IsoRendererOutputChannels; ++i)
			defaults.values[i] = default_values[i];
		output = at::empty({ IsoRendererOutputChannels, height, width },
            at::dtype(at::kFloat).device(at::kCUDA));
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, FillDefaultKernel<IsoRendererOutputChannels>);
		FillDefaultKernel<IsoRendererOutputChannels>
            <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
            (cfg.virtual_size,
                output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
                defaults);
		CUMAT_CHECK_ERROR();
	}
	else if (C == DvrRendererOutputChannels)
	{
		Defaults<DvrRendererOutputChannels> defaults;
		for (int i = 0; i < DvrRendererOutputChannels; ++i)
			defaults.values[i] = default_values[i];
		output = at::empty({ DvrRendererOutputChannels, height, width },
            at::dtype(at::kFloat).device(at::kCUDA));
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, FillDefaultKernel<DvrRendererOutputChannels>);
		FillDefaultKernel<DvrRendererOutputChannels>
            <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
            (cfg.virtual_size,
                output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
                defaults);
		CUMAT_CHECK_ERROR();
	}
	else
	{
		AT_ASSERTM(false, "only IsoRendererOutputChannels or DvrRendererOutputChannels supported as channel count");
	}

	//scatter points
	at::Tensor mask = torch::empty({ 1, height, width }, at::dtype(at::kFloat).device(at::kCUDA));
	scatter_samples_to_image_gpu(sample_position, samples, output, mask, stream);

	return output;
}

END_RENDERER_NAMESPACE
#endif
