#include <stdio.h>
#include "device_launch_parameters.h"
#include "gvdb/cuda_math.cuh"

typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY
#include "gvdb/cuda_gvdb_scene.cuh"		// GVDB Scene
#include "gvdb/cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "gvdb/cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "gvdb/cuda_gvdb_dda.cuh"		// GVDB DDA 
#include "gvdb/cuda_gvdb_raycast.cuh"	// GVDB Raycasting
//--------------------------------

// additional tracing settings
__constant__ int binarySearchSteps = 10;
// light settings
__constant__ float3 lightDir;
__constant__ float3 ambientColor;
__constant__ float3 diffuseColor;
__constant__ float3 specularColor;
__constant__ int specularExponent;
__constant__ float4 currentViewMatrix[4]; //row-major
__constant__ float4 nextViewMatrix[4];
__constant__ float4 normalMatrix[4];
// ambient occlusion
__constant__ int aoSamples;
#define MAX_AMBIENT_OCCLUSION_SAMPLES 512
__constant__ float4 aoHemisphere[MAX_AMBIENT_OCCLUSION_SAMPLES];
#define AMBIENT_OCCLUSION_RANDOM_ROTATIONS 4
__constant__ float4 aoRandomRotations[AMBIENT_OCCLUSION_RANDOM_ROTATIONS * AMBIENT_OCCLUSION_RANDOM_ROTATIONS];
__constant__ float aoRadius; //world space
__constant__ float aoBias = 1e-3; //distance in world space to backtrack the position
                                  //This is needed to reduce acne artifacts
__constant__ int4 viewport; //minX, minY, maxX, maxY

// 0: point sampling, 1: ray sampling
#define AMBIENT_OCCLUSION_MODE 1

inline __host__ __device__ float3 reflect3(float3 i, float3 n)
{
	return i - 2.0f * n * dot(n, i);
}

inline __device__ float4 matmul(const float4 mat[4], float4 v)
{
	return make_float4(
		dot(mat[0], v),
		dot(mat[1], v),
		dot(mat[2], v),
		dot(mat[3], v)
	);
}

__device__ float getValueAtPoint(VDBInfo* gvdb, uchar chan, float3 world_pos)
{
	//find node
	float3 offs, vmin, vdel;
	uint64 node_id;
	VDBNode* node = getNodeAtPoint(gvdb, world_pos, &offs, &vmin, &vdel, &node_id);
	if (node == nullptr) return 0;

	float3 o = offs;				// Atlas sub-volume to trace	
	float3 p = (world_pos - vmin) / gvdb->vdel[0]; // sample point in index coords

	return tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z);
}

#if AMBIENT_OCCLUSION_MODE==0
__device__ float computeAmbientOcclusion(VDBInfo* gvdb, uchar chan, float3 pos, float3 normal, int x, int y)
{
	if (aoSamples == 0) return 1;
	float ao = 0.0;
	//get random rotation vector
	int x2 = x % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
	int y2 = y % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
	float3 noise = make_float3(aoRandomRotations[x2 + AMBIENT_OCCLUSION_RANDOM_ROTATIONS * y2]);
	//compute transformation
	float3 tangent = normalize(noise - normal * dot(noise, normal));
	float3 bitangent = cross(normal, tangent);
	//sample
	float bias = getValueAtPoint(gvdb, chan, pos);
	for (int i=0; i< aoSamples; ++i)
	{
		//get hemisphere sample
		float3 sampleT = make_float3(aoHemisphere[i]);
		//transform to world space
		float3 sampleW = make_float3(
			dot(make_float3(tangent.x, bitangent.x, normal.x), sampleT),
			dot(make_float3(tangent.y, bitangent.y, normal.y), sampleT),
			dot(make_float3(tangent.z, bitangent.z, normal.z), sampleT)
		);
		//query sample
		float3 samplePos = pos + aoRadius * sampleW;
		float value = getValueAtPoint(gvdb, chan, samplePos);
		ao += (value > bias) ? 0.0f : 1.0f;
	}
	ao /= aoSamples;
	return ao;
}

#else

__device__ float computeAmbientOcclusion(VDBInfo* gvdb, uchar chan, float3 pos, float3 normal, int x, int y)
{
	if (aoSamples == 0) return 1;
	float ao = 0.0;
	//get random rotation vector
	int x2 = x % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
	int y2 = y % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
	float3 noise = make_float3(aoRandomRotations[x2 + AMBIENT_OCCLUSION_RANDOM_ROTATIONS * y2]);
	//compute transformation
	float3 tangent = normalize(noise - normal * dot(noise, normal));
	float3 bitangent = cross(normal, tangent);
	//sample
	float bias = getValueAtPoint(gvdb, chan, pos);
	for (int i = 0; i < aoSamples; ++i)
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
		float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
		float4 clr = make_float4(1, 1, 1, 1);
		float3 norm;
		rayCast(gvdb, chan, pos, sampleW, hit, norm, clr, raySurfaceTrilinearBrick);
		float value = 1.0;
		if (hit.z != NOHIT) {
			float dist = length(pos - hit);
			value = smoothstep(1, 0, aoRadius / dist);
		}
		ao += value;
	}
	ao /= aoSamples;
	return ao;
}

#endif

__device__ inline float3 safe_normalize(float3 vec)
{
	float len = length(vec);
	if (!(len > 1e-6)) return make_float3(0.0f);
	return vec / len;
}

// SurfaceTrilinearBrick - Trace brick to render surface with trilinear smoothing
// Custom version with binary search
__device__ void raySurfaceTrilinearBrickCustom(VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr)
{
	float3 vmin;
	VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);	// Get the VDB leaf node	
	float3  o = make_float3(node->mValue);				// Atlas sub-volume to trace	
	float3	p = (pos + t.x*dir - vmin) / gvdb->vdel[0];	// sample point in index coords			
	t.x = SCN_PSTEP * ceil(t.x / SCN_PSTEP);

	float tCurrent = 0;
	float3 pStart = p;

	for (int iter = 0; iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++)
	{
		if (tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z) >= SCN_THRESH) {
#if 1
			//we are inside the volume, perform some binary search steps to find 
			//the closest point *outside* the isosurface
			float tLower = tCurrent - SCN_PSTEP;
			float tUpper = tCurrent;
			for (int i=0; i<binarySearchSteps; ++i)
			{
				float tMiddle = 0.5 * (tLower + tUpper);
				p = pStart + tMiddle * dir;
				if (tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z) >= SCN_THRESH)
					tUpper = tMiddle; //still inside
				else
					tLower = tMiddle; //outside
			}
			p = pStart + tLower * dir;
#endif

			hit = p * gvdb->vdel[0] + vmin;
			norm = getGradient(gvdb, chan, p + o);
			if (gvdb->clr_chan != CHAN_UNDEF) hclr = getColorF(gvdb, gvdb->clr_chan, p + o);
			return;
		}
		p += SCN_PSTEP * dir;
		tCurrent += SCN_PSTEP;
		t.x += SCN_PSTEP;
	}
}

// Custom raycast kernel
extern "C" __global__ void custom_iso_kernel(VDBInfo* gvdb, uchar chan, float* outBuf)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= scn.width || y >= scn.height) return;

	float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 clr = make_float4(1, 1, 1, 1);
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = normalize(getViewRay((float(x) + 0.5) / scn.width, (float(y) + 0.5) / scn.height));

	float3 color = make_float3(0, 0, 0);
	float mask = 0;
	float3 normal = make_float3(0, 0, 0);
	float depth = 0;
	float2 flow = make_float2(0, 0);
	float ao = 1.0f;
	float shadow = 1.0f;

	if (x >= viewport.x && y >= viewport.y && x < viewport.z && y < viewport.w) {

		// Ray march - trace a ray into GVDB and find the closest hit point
		rayCast(gvdb, chan, rpos, rdir, hit, norm, clr, raySurfaceTrilinearBrickCustom);

		if (hit.z != NOHIT) {
			mask = 1;
			//depth = length(rpos - hit);
			norm = safe_normalize(norm);
			//shading
			color = ambientColor;
			float3 eyedir = normalize(rpos - hit);
			color += diffuseColor * abs(dot(norm, lightDir));
			float3 R = normalize(reflect3(lightDir, norm));		// reflection vector
			color += specularColor * ((specularExponent + 2) / (2 * 3.41))
				* pow(max(0.0f, dot(R, eyedir)), specularExponent);
			//flow
			float3 hit2 = mmult(hit, SCN_XFORM);
			float4 xyzw = make_float4(hit2.x, hit2.y, hit2.z, 1.0f);
			float4 screenNext = matmul(nextViewMatrix, xyzw);
			float4 screenCurrent = matmul(currentViewMatrix, xyzw);
			screenCurrent /= screenCurrent.w;
			screenNext /= screenNext.w;
			flow = 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y);
			//depth in screen space
			depth = screenCurrent.z;
			//normal in screen space
			normal = make_float3(matmul(normalMatrix, make_float4(norm, 0)));
			//special shading effects
			ao = computeAmbientOcclusion(gvdb, chan, hit - aoBias * rdir, norm, x, y);
		}
	}
	outBuf[12 * (y*scn.width + x) + 0] = color.x;
	outBuf[12 * (y*scn.width + x) + 1] = color.y;
	outBuf[12 * (y*scn.width + x) + 2] = color.z;
	outBuf[12 * (y*scn.width + x) + 3] = mask;
	outBuf[12 * (y*scn.width + x) + 4] = normal.x;
	outBuf[12 * (y*scn.width + x) + 5] = normal.y;
	outBuf[12 * (y*scn.width + x) + 6] = normal.z;
	outBuf[12 * (y*scn.width + x) + 7] = depth;
	outBuf[12 * (y*scn.width + x) + 8] = flow.x;
	outBuf[12 * (y*scn.width + x) + 9] = flow.y;
	outBuf[12 * (y*scn.width + x) + 10] = ao;
	outBuf[12 * (y*scn.width + x) + 11] = shadow;
}

#if 0
//TODO: unfinished, focus on isosurfaces now

// DeepBrick - Sample into brick for deep volume raytracing
__device__ void rayDeepBrickCustom(VDBInfo* gvdb, uchar chan, int nodeid, 
	float3 t, float3 pos, float3 dir, 
	float3& pstep, 
	float3& hit, //hit.x=first hit, hit.y=mean depth
	float3& norm, float4& clr)
{
	float3 vmin;
	VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);			// Get the VDB leaf node		

	//t.x = SCN_PSTEP * ceil( t.x / SCN_PSTEP );						// start on sampling wavefront	

	float3 o = make_float3(node->mValue);					// atlas sub-volume to trace
	float3 wp = pos + t.x*dir;
	float3 p = (wp - vmin) / gvdb->vdel[0];					// sample point in index coords	
	float3 wpt = SCN_PSTEP * dir * gvdb->vdel[0];					// world increment
	float4 val = make_float4(0, 0, 0, 0);
	float4 hclr;
	int iter = 0;
	float dt = length(SCN_PSTEP*dir*gvdb->vdel[0]);

	// record front hit point at first significant voxel
	if (hit.x == 0) {
		hit.x = t.x; // length(wp - pos);
		hit.y = 0; //weight accumulator for the weighted mean depth
	}

	// skip empty voxels
	for (iter = 0; val.w < SCN_MINVAL && iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) {
		val.w = transfer(gvdb, tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z)).w;
		p += SCN_PSTEP * dir;
		wp += wpt;
		t.x += dt;
	}

	// accumulate remaining voxels
	for (; clr.w > SCN_ALPHACUT && iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) {

		// depth buffer test [optional]
		if (SCN_DBUF != 0x0) {
			if (t.x > getLinearDepth(SCN_DBUF)) {
				hit.y = length(wp - pos);
				hit.z = 1;
				clr = make_float4(fmin(clr.x, 1.f), fmin(clr.y, 1.f), fmin(clr.z, 1.f), fmax(clr.w, 0.f));
				return;
			}
		}
		val = transfer(gvdb, tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z));
		val.w = exp(SCN_EXTINCT * val.w * SCN_PSTEP);

		hit.y += t.x * (1 - val.w) * SCN_ALBEDO * hclr.z;

		hclr = (gvdb->clr_chan == CHAN_UNDEF) ? make_float4(1, 1, 1, 1) : getColorF(gvdb, gvdb->clr_chan, p + o);
		clr.x += val.x * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.x;
		clr.y += val.y * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.y;
		clr.z += val.z * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.z;
		clr.w *= val.w;

		p += SCN_PSTEP * dir;
		wp += wpt;
		t.x += dt;
	}
	clr = make_float4(fmin(clr.x, 1.f), fmin(clr.y, 1.f), fmin(clr.z, 1.f), fmax(clr.w, 0.f));
}


extern "C" __global__ void custom_raycast_kernel(VDBInfo* gvdb, uchar chan, float* outBuf)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= scn.width || y >= scn.height) return;

	float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 color = make_float4(0, 0, 0, 1);
	float3 normal = make_float3(0, 0, 0);
	float3 rpos = getViewPos();
	float3 rdir = normalize(getViewRay((float(x) + 0.5) / scn.width, (float(y) + 0.5) / scn.height));

	rayCast(gvdb, chan, rpos, rdir, hit, normal, color, rayDeepBrickCustom);

	float mask = 1 - color.w;
	float depth = length(rpos - hit);
	float2 flow = make_float2(0, 0);

	outBuf[10 * (y*scn.width + x) + 0] = color.x;
	outBuf[10 * (y*scn.width + x) + 1] = color.y;
	outBuf[10 * (y*scn.width + x) + 2] = color.z;
	outBuf[10 * (y*scn.width + x) + 3] = mask;
	outBuf[10 * (y*scn.width + x) + 4] = normal.x;
	outBuf[10 * (y*scn.width + x) + 5] = normal.y;
	outBuf[10 * (y*scn.width + x) + 6] = normal.z;
	outBuf[10 * (y*scn.width + x) + 7] = depth;
	outBuf[10 * (y*scn.width + x) + 8] = flow.x;
	outBuf[10 * (y*scn.width + x) + 9] = flow.y;
}
#endif