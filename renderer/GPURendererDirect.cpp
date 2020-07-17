// GPURenderer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <CLI11.hpp>
#include <tinyformat.h>
#include <boost/algorithm/string/predicate.hpp>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>

#include <gvdb/gvdb.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <fcntl.h>
#include <io.h>

#include "ShadowFX.h"
#include "Vdb2Vbx.h"

using namespace nvdb;

VolumeGVDB gvdb;
std::unique_ptr<Camera3D> camera;
CUmodule cuCustom;
CUfunction cuIsoKernel;
CUdeviceptr kernelLightDir;
CUdeviceptr kernelAmbientColor;
CUdeviceptr kernelDiffuseColor;
CUdeviceptr kernelSpecularColor;
CUdeviceptr kernelSpecularExponent;
CUdeviceptr kernelCurrentViewMatrix;
CUdeviceptr kernelNextViewMatrix;
CUdeviceptr kernelNormalMatrix;
CUdeviceptr kernelAoSamples;
CUdeviceptr kernelAoHemisphere;
CUdeviceptr kernelAoRandomRotations;
CUdeviceptr kernelAoRadius;
CUdeviceptr kernelViewport;
#define MAX_AMBIENT_OCCLUSION_SAMPLES 512  //same as in render_kernel.cu
#define AMBIENT_OCCLUSION_RANDOM_ROTATIONS 4

namespace nvdb {
	std::ostream& operator<<(std::ostream& o, const Vector3DF& v)
	{
		o << v.x << "," << v.y << "," << v.z;
		return o;
	}
}

template<typename Arg>
void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg)
{
	if (str.find(',', pos) != std::string::npos)
		throw std::exception("more list entries specified than expected");
	std::stringstream ss(str.substr(pos));
	ss >> arg;
}

template<typename Arg, typename... Rest>
void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg, Rest& ... rest)
{
	size_t p = str.find(',', pos);
	if (p == std::string::npos)
		throw std::exception("less list entries specified than expected");
	std::stringstream ss(str.substr(pos, p));
	ss >> arg;
	parseStrImpl(str, p + 1, rest...);
}

template<typename ...Args>
void parseStr(const std::string& str, Args & ... args)
{
	parseStrImpl(str, 0, args...);
}

//Does the string contain a '%d' or similiar?
bool hasIntFormat(const std::string& str)
{
	try
	{
		int index = 1;
		tfm::format(str, index);
		return true;
	}
	catch (std::exception& ex)
	{
		return false;
	}
}

struct Args
{
	//input
	std::string inputFilename;
	//camera
	int resolutionX = 512;
	int resolutionY = 512;
	double cameraFov = 45;
	Vector3DF cameraOrigin = { 0,0,-1 };
	Vector3DF cameraLookAt = { 0,0,0 };
	Vector3DF cameraUp = { 0,1,0 };
	int noShading = 0;
	int4 viewport = make_int4(0, 0, resolutionX, resolutionY);
	//isosurface
	double isovalue = 0.0;
	Vector3DF materialDiffuse = { 0.7, 0.7, 0.7 };
	Vector3DF materialSpecular = { 1,1,1 };
	Vector3DF materialAmbient = { 0.1,0.1,0.1 };
	int materialSpecularExponent = 32;
	bool cameraLight = true; //true: light along camera view direction
	Vector3DF lightDirection = { 0,0,0 };
	int samples = 1;
	//iso effects
	int aoSamples = 32;
	float aoRadius = 0.01;
};
Args args;
Vector3DF lastOrigin;
Vector3DF lastLookAt;

bool cudaCheck(CUresult status, const char* msg)
{
	if (status != CUDA_SUCCESS) {
		const char* stat = "";
		cuGetErrorString(status, &stat);
		std::cout << "CUDA ERROR: " << stat << "(in " << msg << ")" << std::endl;
		exit(-1);
		return false;
	}
	return true;
}

//Computes ambient occlusion parameters
void computeAmbientOcclusionParameters(const Args& args)
{
	static std::default_random_engine rnd;
	static std::uniform_real_distribution<float> distr(0.0f, 1.0f);
	//samples
	int samples = args.aoSamples;
	samples = std::max(0, std::min(MAX_AMBIENT_OCCLUSION_SAMPLES - 1, samples));
	cudaCheck(cuMemcpyHtoD(kernelAoSamples, &samples, sizeof(int)), "cuMemcpyHtoD");
	//radius
	cudaCheck(cuMemcpyHtoD(kernelAoRadius, &args.aoRadius, sizeof(float)), "cuMemcpyHtoD");
	//samples on a hemisphere
	std::vector<float4> aoHemisphere(MAX_AMBIENT_OCCLUSION_SAMPLES, make_float4(0, 0, 0, 0));
	for (int i = 0; i < MAX_AMBIENT_OCCLUSION_SAMPLES; ++i)
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
	cudaCheck(cuMemcpyHtoD(
		kernelAoHemisphere,
		aoHemisphere.data(),
		sizeof(float) * 4 * MAX_AMBIENT_OCCLUSION_SAMPLES),
		"cuMemcpyHtoD");
	//random rotation vectors
	std::vector<float4> aoRandomRotations(AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS);
	for (int i = 0; i < AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS; ++i)
	{
		float x = distr(rnd) * 2 - 1;
		float y = distr(rnd) * 2 - 1;
		float linv = 1.0f / sqrt(x*x + y * y);
		aoRandomRotations[i] = make_float4(x*linv, y*linv, 0, 0);
	}
	cudaCheck(cuMemcpyHtoD(
		kernelAoRandomRotations,
		aoRandomRotations.data(),
		sizeof(float) * 4 * AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS),
		"cuMemcpyHtoD");
}

void initRendering(const Args& args)
{
	Scene* scn = gvdb.getScene();
	//camera
	camera = std::make_unique<Camera3D>();
	camera->setFov(args.cameraFov);
	camera->setAspect(float(args.resolutionX) / float(args.resolutionY));
	camera->setPos(args.cameraOrigin.x, args.cameraOrigin.y, args.cameraOrigin.z);
	camera->setToPos(args.cameraLookAt.x, args.cameraLookAt.y, args.cameraLookAt.z);
	scn->SetCamera(camera.get());
	scn->SetRes(args.resolutionX, args.resolutionY);
	//output
	gvdb.AddRenderBuf(0, args.resolutionX, args.resolutionY, 12 * sizeof(float));

	//load rendering kernel
	std::cout << "Load rendering kernel" << std::endl;
	cudaCheck(cuModuleLoad(&cuCustom, "render_kernel.ptx"), "cuModuleLoad (render_custom)");
	cudaCheck(cuModuleGetFunction(&cuIsoKernel, cuCustom, "custom_iso_kernel"), "cuModuleGetFunction (raycast_kernel)");
	gvdb.SetModule(cuCustom);

	//grap rendering settings
	cudaCheck(cuModuleGetGlobal(&kernelLightDir, NULL, cuCustom, "lightDir"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAmbientColor, NULL, cuCustom, "ambientColor"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelDiffuseColor, NULL, cuCustom, "diffuseColor"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelSpecularColor, NULL, cuCustom, "specularColor"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelSpecularExponent, NULL, cuCustom, "specularExponent"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelCurrentViewMatrix, NULL, cuCustom, "currentViewMatrix"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelNextViewMatrix, NULL, cuCustom, "nextViewMatrix"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelNormalMatrix, NULL, cuCustom, "normalMatrix"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoSamples, NULL, cuCustom, "aoSamples"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoHemisphere, NULL, cuCustom, "aoHemisphere"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoRandomRotations, NULL, cuCustom, "aoRandomRotations"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoRadius, NULL, cuCustom, "aoRadius"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelViewport, NULL, cuCustom, "viewport"), "cuModuleGetGlobal");
	computeAmbientOcclusionParameters(args);
}

extern "C" {
	__declspec(dllexport) int initGVDB()
	{
		std::cout << "Start GVDB" << std::endl;
#ifdef NDEBUG
		gvdb.SetDebug(false);
#else
		gvdb.SetDebug(true);
#endif
		gvdb.SetVerbose(true);
		gvdb.SetProfile(false, true);
		gvdb.SetCudaDevice(GVDB_DEV_FIRST);
		gvdb.Initialize();

		initRendering(args);

		return 0;
	}
}

extern "C" {
	__declspec(dllexport) int loadGrid(const char* filename)
	{
		gvdb.Clear();
		gvdb.ClearAllChannels();

		std::string filenameStr(filename);
		if (!boost::ends_with(filenameStr, ".vbx"))
		{
			std::cout << "Error: Input must end in .vbx" << std::endl;
			return -1;
		}
		std::cout << "Load " << filenameStr << std::endl;
		if (!gvdb.LoadVBX(filenameStr)) {
			std::cout << "Unable to load VBX file" << std::endl;
			return -2;
		}

		//grid statistics and transformation
		Vector3DF objmin, objmax, voxmin, voxmax, voxsize, voxres;
		gvdb.getDimensions(objmin, objmax, voxmin, voxmax, voxsize, voxres);
		std::cout << "objmin=" << objmin << std::endl;
		std::cout << "objmax=" << objmax << std::endl;
		std::cout << "voxmin=" << voxmin << std::endl;
		std::cout << "voxmax=" << voxmax << std::endl;
		std::cout << "voxsize=" << voxsize << std::endl;
		std::cout << "voxres=" << voxres << std::endl;
		Vector3DF invcenter = (objmax + objmin) * -0.5f;
		float scale = 0.5 / std::max({ objmax.x - objmin.x, objmax.y - objmin.y, objmax.z - objmin.z });
		gvdb.SetTransform(invcenter, Vector3DF(scale, scale, scale), Vector3DF(0, 0, 0), Vector3DF(0, 0, 0));

		//reset camera
		lastOrigin = args.cameraOrigin;
		lastLookAt = args.cameraLookAt;

		return 0;
	}
}

std::string getPointerType(const void* ptr)
{
	cudaPointerAttributes attr = {};
	cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
	if (err != cudaSuccess)
		return "ERROR!";
	if (attr.type == cudaMemoryTypeDevice)
		return "device";
	else if (attr.type == cudaMemoryTypeHost)
		return "host";
	else if (attr.type == cudaMemoryTypeManaged)
		return "managed";
	else if (attr.type == cudaMemoryTypeUnregistered)
		return "unregistered";
	else
		return "unknown";
}
std::string getPointerType(CUdeviceptr ptr)
{
	return getPointerType(reinterpret_cast<void*>(ptr));
}

bool render(const Args& args,
	const Vector3DF& currentOrigin, const Vector3DF& currentLookAt,
	const Vector3DF& nextOrigin, const const Vector3DF& nextLookAt,
	CUdeviceptr targetDevice, // output is copied into this memory
	float* secondsOut = nullptr)
{
	//std::cout << "Entry: render" << std::endl;
	Scene* scn = gvdb.getScene();
	bool resize = args.resolutionX != scn->getRes().x || args.resolutionY != scn->getRes().y;
	//camera
	camera->setFov(args.cameraFov);
	camera->setAspect(float(args.resolutionX) / float(args.resolutionY));
	camera->up_dir = args.cameraUp;
	camera->setPos(currentOrigin.x, currentOrigin.y, currentOrigin.z);
	camera->setToPos(currentLookAt.x, currentLookAt.y, currentLookAt.z);
	scn->SetRes(args.resolutionX, args.resolutionY);
	Matrix4F modelViewProj = camera->getFullProjMatrix(); 
	modelViewProj *= camera->getViewMatrix();
	modelViewProj.Transpose();
	cudaCheck(cuMemcpyHtoD(kernelCurrentViewMatrix, modelViewProj.data, sizeof(float) * 16), "cuMemcpyHtoD");
	Camera3D nextCamera = *camera; nextCamera.Copy(*camera);
	nextCamera.setPos(nextOrigin.x, nextOrigin.y, nextOrigin.z);
	nextCamera.setToPos(nextLookAt.x, nextLookAt.y, nextLookAt.z);
	modelViewProj = nextCamera.getFullProjMatrix(); 
	modelViewProj *= nextCamera.getViewMatrix();
	modelViewProj.Transpose();
	cudaCheck(cuMemcpyHtoD(kernelNextViewMatrix, modelViewProj.data, sizeof(float) * 16), "cuMemcpyHtoD");
	Matrix4F normalMatrix = camera->getViewMatrix();
	normalMatrix.Transpose();
	cudaCheck(cuMemcpyHtoD(kernelNormalMatrix, normalMatrix.data, sizeof(float) * 16), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelViewport, &args.viewport, sizeof(int)*4), "cuMemcpyHtoD");
	//light and color
	Vector3DF lightDir = args.cameraLight
		? (Vector3DF(args.cameraLookAt) - args.cameraOrigin)
		: args.lightDirection;
	lightDir.Normalize();
	cudaCheck(cuMemcpyHtoD(kernelLightDir, &lightDir.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelAmbientColor, &args.materialAmbient.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelDiffuseColor, &args.materialDiffuse.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelSpecularColor, &args.materialSpecular.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelSpecularExponent, &args.materialSpecularExponent, sizeof(int)), "cuMemcpyHtoD");
	//ambient occlusion
	//samples
	int aoSamples = args.aoSamples;
	aoSamples = std::max(0, std::min(MAX_AMBIENT_OCCLUSION_SAMPLES, aoSamples));
	cudaCheck(cuMemcpyHtoD(kernelAoSamples, &aoSamples, sizeof(int)), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelAoRadius, &args.aoRadius, sizeof(float)), "cuMemcpyHtoD");

	//output
	if (resize)
		gvdb.ResizeRenderBuf(0, args.resolutionX, args.resolutionY, 12 * sizeof(float));

	//std::cout << "Settings done" << std::endl;

	//iso and step
	scn->SetVolumeRange(args.isovalue, 0.0f, 1.0f);
	scn->SetSteps(0.05, 16, 0.05);

	//render
	cudaCheck(cuCtxSynchronize(), "cudaDeviceSynchronize()");
	auto start = std::chrono::high_resolution_clock::now();
	gvdb.RenderKernel(cuIsoKernel, 0, 0);

	//grab output
	//std::cout << "source pointer: 0x" << std::hex << gvdb.mRenderBuf[0].gpu
	//	<< " (" << getPointerType(gvdb.mRenderBuf[0].gpu) << ")" << std::endl;
	//std::cout << "target pointer: 0x" << std::hex << targetDevice
	//	<< " (" << getPointerType(targetDevice) << ")" << std::endl;
	size_t num_bytes = args.resolutionX * args.resolutionY * 12 * sizeof(float);
	cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(targetDevice),
		reinterpret_cast<const void*>(gvdb.mRenderBuf[0].gpu),
		num_bytes, cudaMemcpyDeviceToDevice);
	if (err == cudaErrorInvalidValue)
		std::cout << "Invalid value" << std::endl;
	
	cudaCheck(cuCtxSynchronize(), "cudaDeviceSynchronize()");
	auto finish = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<double>(finish - start).count();
	if (secondsOut) *secondsOut = time;
	//std::cout << "Rendering done: " << time << std::endl;

	return true;
}

extern "C"
{
	__declspec(dllexport) int setParameter(const char* cmd_, const char* value_)
	{
		std::string cmd(cmd_);
		std::string value(value_);
		if (cmd == "fov")
			parseStr(value, args.cameraFov);
		else if (cmd == "cameraOrigin")
			parseStr(value, args.cameraOrigin.x, args.cameraOrigin.y, args.cameraOrigin.z);
		else if (cmd == "cameraLookAt")
			parseStr(value, args.cameraLookAt.x, args.cameraLookAt.y, args.cameraLookAt.z);
		else if (cmd == "cameraUp")
			parseStr(value, args.cameraUp.x, args.cameraUp.y, args.cameraUp.z);
		else if (cmd == "cameraFoV")
			parseStr(value, args.cameraFov);
		else if (cmd == "resolution")
			parseStr(value, args.resolutionX, args.resolutionY);
		else if (cmd == "isovalue")
			parseStr(value, args.isovalue);
		else if (cmd == "unshaded")
			parseStr(value, args.noShading);
		else if (cmd == "aosamples")
			parseStr(value, args.aoSamples);
		else if (cmd == "aoradius")
			parseStr(value, args.aoRadius);
		else if (cmd == "viewport") //minX, minY, maxX, maxY
			parseStr(value, args.viewport.x, args.viewport.y, args.viewport.z, args.viewport.w);
		//TODO: more commands
		else {
			std::cout << "Unknown command: '" << cmd << "', exit" << std::endl;
			return -1;
		}
		return 0;
	}
}

extern "C"
{
	__declspec(dllexport) float render(CUdeviceptr targetTensor)
	{
		float time;
		//std::cout << "Render to " << targetTensor << std::endl;
		bool success = render(args,
			args.cameraOrigin, args.cameraLookAt,
			lastOrigin, lastLookAt, targetTensor, &time);
		//std::cout << "success: " << success << ", time: " << time << std::endl;
		//save old camera position for flow
		lastOrigin = args.cameraOrigin;
		lastLookAt = args.cameraLookAt;
		//return time
		return success ? time : -1;
	}
}
