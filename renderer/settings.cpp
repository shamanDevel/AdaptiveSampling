#include "settings.h"
#include <stdexcept>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

RendererArgs TheRendererArgs;

template<typename Arg>
static void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg)
{
	if (str.find(',', pos) != std::string::npos)
		throw std::runtime_error("more list entries specified than expected");
	std::stringstream ss(str.substr(pos));
	ss >> arg;
}

template<typename Arg, typename... Rest>
static void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg, Rest& ... rest)
{
	size_t p = str.find(',', pos);
	if (p == std::string::npos)
		throw std::runtime_error("less list entries specified than expected");
	std::stringstream ss(str.substr(pos, p));
	ss >> arg;
	parseStrImpl(str, p + 1, rest...);
}

template<typename ...Args>
static void parseStr(const std::string& str, Args & ... args)
{
	parseStrImpl(str, 0, args...);
}

template<typename T>
static void parseStr(const std::string& str, std::vector<T>& vec)
{
	std::stringstream ss(str);

	T i;
	while (ss >> i)
	{
		vec.push_back(i);

		if (ss.peek() == ',')
		{
			ss.ignore();
		}
	}
}

int64_t SetRendererParameter(const std::string& cmd, const std::string& value)
{
	if (cmd == "mipmapLevel")
		parseStr(value, TheRendererArgs.mipmapLevel);
	else if (cmd == "fov")
		parseStr(value, TheRendererArgs.cameraFovDegrees);
	else if (cmd == "cameraOrigin")
		parseStr(value, TheRendererArgs.cameraOrigin.x, TheRendererArgs.cameraOrigin.y, TheRendererArgs.cameraOrigin.z);
	else if (cmd == "cameraLookAt")
		parseStr(value, TheRendererArgs.cameraLookAt.x, TheRendererArgs.cameraLookAt.y, TheRendererArgs.cameraLookAt.z);
	else if (cmd == "cameraUp")
		parseStr(value, TheRendererArgs.cameraUp.x, TheRendererArgs.cameraUp.y, TheRendererArgs.cameraUp.z);
	else if (cmd == "cameraFoV")
		parseStr(value, TheRendererArgs.cameraFovDegrees);
	else if (cmd == "resolution")
		parseStr(value, TheRendererArgs.cameraResolutionX, TheRendererArgs.cameraResolutionY);
	else if (cmd == "isovalue")
		parseStr(value, TheRendererArgs.isovalue);
	else if (cmd == "aosamples")
		parseStr(value, TheRendererArgs.aoSamples);
	else if (cmd == "aoradius")
		parseStr(value, TheRendererArgs.aoRadius);
	else if (cmd == "viewport") //minX, minY, maxX, maxY
		parseStr(value, TheRendererArgs.cameraViewport.x, TheRendererArgs.cameraViewport.y, TheRendererArgs.cameraViewport.z, TheRendererArgs.cameraViewport.w);
	else if (cmd == "binarySearchSteps")
		parseStr(value, TheRendererArgs.binarySearchSteps);
	else if (cmd == "stepsize")
		parseStr(value, TheRendererArgs.stepsize);
	else if (cmd == "interpolation") {
		int option = 0;
		parseStr(value, option);
		TheRendererArgs.volumeFilterMode = (RendererArgs::VolumeFilterMode)(option);
	}
	else if (cmd == "renderMode") {
		int option = 0;
		parseStr(value, option);
		TheRendererArgs.renderMode = (RendererArgs::RenderMode)(option);
	}
	else if (cmd == "densityAxisOpacity") {
		TheRendererArgs.densityAxisOpacity.clear();
		parseStr(value, TheRendererArgs.densityAxisOpacity);
	}
	else if (cmd == "opacityAxis") {
		TheRendererArgs.opacityAxis.clear();
		parseStr(value, TheRendererArgs.opacityAxis);
	}
	else if (cmd == "densityAxisColor") {
		TheRendererArgs.densityAxisColor.clear();
		parseStr(value, TheRendererArgs.densityAxisColor);
	}
	else if (cmd == "colorAxis") {
		TheRendererArgs.colorAxis.clear();
		std::vector<float> colorAxisTemp;
		parseStr(value, colorAxisTemp);

		int size = colorAxisTemp.size() / 3;
		for (int i = 0; i < size; ++i)
		{
			TheRendererArgs.colorAxis.push_back(make_float3(colorAxisTemp[i * 3], colorAxisTemp[i * 3 + 1], colorAxisTemp[i * 3 + 2]));
		}
	}
	else if (cmd == "minDensity") {
		parseStr(value, TheRendererArgs.minDensity);
	}
	else if (cmd == "maxDensity") {
		parseStr(value, TheRendererArgs.maxDensity);
	}
	else if (cmd == "opacityScaling") {
		parseStr(value, TheRendererArgs.opacityScaling);
	}
	else if (cmd == "ambientLightColor")
	{
		parseStr(value,
			TheRendererArgs.shading.ambientLightColor.x,
			TheRendererArgs.shading.ambientLightColor.y,
			TheRendererArgs.shading.ambientLightColor.z);
	}
	else if (cmd == "diffuseLightColor")
	{
		parseStr(value,
			TheRendererArgs.shading.diffuseLightColor.x,
			TheRendererArgs.shading.diffuseLightColor.y,
			TheRendererArgs.shading.diffuseLightColor.z);
	}
	else if (cmd == "specularLightColor")
	{
		parseStr(value,
			TheRendererArgs.shading.specularLightColor.x,
			TheRendererArgs.shading.specularLightColor.y,
			TheRendererArgs.shading.specularLightColor.z);
	}
	else if (cmd == "specularExponent")
	{
		parseStr(value, TheRendererArgs.shading.specularExponent);
	}
	else if (cmd == "materialColor")
	{
		parseStr(value,
			TheRendererArgs.shading.materialColor.x,
			TheRendererArgs.shading.materialColor.y,
			TheRendererArgs.shading.materialColor.z);
	}
	else if (cmd == "lightDirection")
	{
		parseStr(value,
			TheRendererArgs.shading.lightDirection.x,
			TheRendererArgs.shading.lightDirection.y,
			TheRendererArgs.shading.lightDirection.z);
	}
	else if (cmd == "dvrUseShading")
	{
		int val;
		parseStr(value, val);
		TheRendererArgs.dvrUseShading = val > 0;
	}
	else {
		std::cerr << "Unknown command: '" << cmd << "', exit" << std::endl;
		return -1;
	}

	return +1;
}

END_RENDERER_NAMESPACE
#endif
