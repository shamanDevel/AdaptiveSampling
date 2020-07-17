#include "visualizer.h"

#include <experimental/filesystem>

#include <lib.h>
#include <cuMat/src/Errors.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/grad_mode.h>

#define NOMINMAX
#include <windows.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "imgui/imgui.h"
#include "imgui/IconsFontAwesome5.h"
#include "imgui/imgui_extension.h"
#include "imgui/imgui_internal.h"

#include <json.hpp>
#include <lodepng.h>
#include <portable-file-dialogs.h>
#include "utils.h"
#include "hsv_normalization.h"

const char* Visualizer::RedrawModeNames[] = {
	"None", "Foveated", "Post", "Network", "Renderer"
};
const char* Visualizer::ChannelModeNames[] = {
	"Mask", "Normal", "Depth", "AO", "Flow", "Color"
};
const char* Visualizer::FoveatedBlurShapeNames[] = {
	"linear", "smoothstep"
};
bool VisualizerEvaluateNetworksInHalfPrecision = false;

Visualizer::Visualizer(GLFWwindow* window)
	: window_(window)
{
	//pre-populate super-resolution networks
	const int isoOutputChannels = 6; //mask, normal x,y,z, depth, ao
	networksIso_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationNearest, 4, isoOutputChannels));
	networksIso_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationLinear, 4, isoOutputChannels));
	networksIso_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationBicubic, 4, isoOutputChannels));
	networksIso_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationNearest, 1, isoOutputChannels));
	for (auto& n : networksIso_)
		n->canDelete = false;

	const int dvrOutputChannels = 8; //red, green, blue, alpha, normal x,y,z, depth
	networksDvr_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationNearest, 4, dvrOutputChannels));
	networksDvr_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationLinear, 4, dvrOutputChannels));
	networksDvr_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationBicubic, 4, dvrOutputChannels));
	networksDvr_.push_back(BaselineSuperresolution::GetMethod(
		BaselineSuperresolution::InterpolationNearest, 1, dvrOutputChannels));
	for (auto& n : networksDvr_)
		n->canDelete = false;

	//pre-populate importance sample networks
	importanceSamplerIso_.push_back(std::make_shared<ImportanceSamplingConstant>());
	importanceSamplerIso_.push_back(std::make_shared<ImportanceSamplingNormalGrad>(1, 3));
	for (auto& n : importanceSamplerIso_)
		n->canDelete = false;
	importanceSamplerDvr_.push_back(std::make_shared<ImportanceSamplingConstant>());
	importanceSamplerDvr_.push_back(std::make_shared<ImportanceSamplingNormalGrad>(0, 3));
	for (auto& n : importanceSamplerDvr_)
		n->canDelete = false;
	stepsizeImportanceSamplerDvr_.push_back(std::make_shared<ImportanceSamplingConstant>());
	stepsizeImportanceSamplerDvr_.push_back(std::make_shared<ImportanceSamplingNormalGrad>(0, 3));
	for (auto& n : stepsizeImportanceSamplerDvr_)
		n->canDelete = false;

	//pre-populate sparse networks
	sparseNetworksIso_.push_back(std::make_shared<AdaptiveReconstructionGroundTruth>());
	sparseNetworksIso_.push_back(std::make_shared<AdaptiveReconstructionMask>(IsosurfaceRendering));
	sparseNetworksIso_.push_back(std::make_shared<AdaptiveReconstructionInterpolated>());
	for (auto& n : sparseNetworksIso_)
		n->canDelete = false;
	sparseNetworksDvr_.push_back(std::make_shared<AdaptiveReconstructionGroundTruth>());
	sparseNetworksDvr_.push_back(std::make_shared<AdaptiveReconstructionMask>(DirectVolumeRendering));
	sparseNetworksDvr_.push_back(std::make_shared<AdaptiveReconstructionInterpolated>());
	for (auto& n : sparseNetworksDvr_)
		n->canDelete = false;
	stepsizeReconstructionDvr_.push_back(std::make_shared<AdaptiveStepsizeReconstructionGroundTruth>());
	stepsizeReconstructionDvr_.push_back(std::make_shared<AdaptiveStepsizeReconstructionBaseline>(DirectVolumeRendering));
	for (auto& n : stepsizeReconstructionDvr_)
		n->canDelete = false;

	// Add .ini handle for ImGuiWindow type
	ImGuiSettingsHandler ini_handler;
	ini_handler.TypeName = "Visualizer";
	ini_handler.TypeHash = ImHashStr("Visualizer");
	static const auto replaceWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == ' ') cpy[i] = '%'; //'%' is not allowed in path names
		return cpy;
	};
	static const auto insertWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == '%') cpy[i] = ' '; //'%' is not allowed in path names
		return cpy;
	};
	auto settingsReadOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void*
	{
		return handler->UserData;
	};
	auto settingsReadLine = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		char path[MAX_PATH];
		int intValue = 0;
		memset(path, 0, sizeof(char)*MAX_PATH);
		std::cout << "reading \"" << line << "\"" << std::endl;
		if (sscanf(line, "VolumeDir=%s", path) == 1)
			vis->volumeDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "TfDir=%s", path) == 1)
			vis->tfDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "NetworkDirIso=%s", path) == 1)
			vis->networkDirectoryIso_ = insertWhitespace(std::string(path));
		if (sscanf(line, "NetworkDirDvr=%s", path) == 1)
			vis->networkDirectoryDvr_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SamplingPatternDir=%s", path) == 1)
			vis->samplingPatternDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SparseNetworkDirIso=%s", path) == 1)
			vis->sparseNetworkDirectoryIso_ = insertWhitespace(std::string(path));
		if (sscanf(line, "ImportanceNetDirIso=%s", path) == 1)
			vis->importanceNetDirectoryIso_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SparseNetworkDirDvr=%s", path) == 1)
			vis->sparseNetworkDirectoryDvr_ = insertWhitespace(std::string(path));
		if (sscanf(line, "ImportanceNetDirDvr=%s", path) == 1)
			vis->importanceNetDirectoryDvr_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SamplingSequenceDir=%s", path) == 1)
			vis->samplingSequenceDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "StepsizeImportanceNetDirDvr=%s", path) == 1)
			vis->stepsizeImportanceDirectoryDvr_ = insertWhitespace(std::string(path));
		if (sscanf(line, "StepsizeReconstructionNetDirDvr=%s", path) == 1)
			vis->stepsizeReconstructionDirectoryDvr_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsDir=%s", path) == 1)
			vis->settingsDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsToLoad=%d", &intValue) == 1)
			vis->settingsToLoad_ = intValue;
	};
	auto settingsWriteAll = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		buf->reserve(200);
		buf->appendf("[%s][Settings]\n", handler->TypeName);
		std::string volumeDir = replaceWhitespace(vis->volumeDirectory_);
		std::string tfDir = replaceWhitespace(vis->tfDirectory_);
		std::string networkDirIso = replaceWhitespace(vis->networkDirectoryIso_);
		std::string networkDirDvr = replaceWhitespace(vis->networkDirectoryDvr_);
		std::string samplingPatternDirectory = replaceWhitespace(vis->samplingPatternDirectory_);
		std::string sparseNetworkDirectoryIso = replaceWhitespace(vis->sparseNetworkDirectoryIso_);
		std::string importanceNetDirectoryIso = replaceWhitespace(vis->importanceNetDirectoryIso_);
		std::string sparseNetworkDirectoryDvr = replaceWhitespace(vis->sparseNetworkDirectoryDvr_);
		std::string importanceNetDirectoryDvr = replaceWhitespace(vis->importanceNetDirectoryDvr_);
		std::string samplingSequenceDirectory = replaceWhitespace(vis->samplingSequenceDirectory_);
		std::string stepsizeImportanceDirectoryDvr = replaceWhitespace(vis->stepsizeImportanceDirectoryDvr_);
		std::string stepsizeReconstructionDirectoryDvr = replaceWhitespace(vis->stepsizeReconstructionDirectoryDvr_);
		std::string settingsDirectory = replaceWhitespace(vis->settingsDirectory_);
		std::cout << "Write settings:" << std::endl;
		buf->appendf("VolumeDir=%s\n", volumeDir.c_str());
		buf->appendf("TfDir=%s\n", tfDir.c_str());
		buf->appendf("NetworkDirIso=%s\n", networkDirIso.c_str());
		buf->appendf("NetworkDirDvr=%s\n", networkDirDvr.c_str());
		buf->appendf("SamplingPatternDir=%s\n", samplingPatternDirectory.c_str());
		buf->appendf("SparseNetworkDirIso=%s\n", sparseNetworkDirectoryIso.c_str());
		buf->appendf("ImportanceNetDirIso=%s\n", importanceNetDirectoryIso.c_str());
		buf->appendf("SparseNetworkDirDvr=%s\n", sparseNetworkDirectoryDvr.c_str());
		buf->appendf("ImportanceNetDirDvr=%s\n", importanceNetDirectoryDvr.c_str());
		buf->appendf("SamplingSequenceDir=%s\n", samplingSequenceDirectory.c_str());
		buf->appendf("StepsizeImportanceNetDirDvr=%s\n", stepsizeImportanceDirectoryDvr.c_str());
		buf->appendf("StepsizeReconstructionNetDirDvr=%s\n", stepsizeReconstructionDirectoryDvr.c_str());
		buf->appendf("SettingsDir=%s\n", settingsDirectory.c_str());
		buf->appendf("SettingsToLoad=%d\n", vis->settingsToLoad_);
		buf->appendf("\n");
	};
	ini_handler.UserData = this;
	ini_handler.ReadOpenFn = settingsReadOpen;
	ini_handler.ReadLineFn = settingsReadLine;
	ini_handler.WriteAllFn = settingsWriteAll;
	GImGui->SettingsHandlers.push_back(ini_handler);

	//pre-fill foveated layers
	foveatedLayers_.push_back(FoveatedLayerDesc{
		{},
		BaselineSuperresolution::GetMethod(BaselineSuperresolution::InterpolationNearest, 1, 6),
		20 });
	foveatedLayers_.push_back(FoveatedLayerDesc{
		{},
		BaselineSuperresolution::GetMethod(BaselineSuperresolution::InterpolationNearest, 2, 6),
		40 });
	foveatedLayers_.push_back(FoveatedLayerDesc{
		{},
		nullptr,
		60 });
	foveatedLayers_.push_back(FoveatedLayerDesc{
		{},
		BaselineSuperresolution::GetMethod(BaselineSuperresolution::InterpolationNearest, 4, 6),
		100 });

	//initialize renderer
	renderer::initializeRenderer();

	//in background, initialize PyTorch
	worker_.launch([](BackgroundWorker* w)
		{
			kernel::initializePyTorch();
			HsvNormalization::Instance();
			std::cout << "PyTorch initialized" << std::endl;
		});
}

Visualizer::~Visualizer()
{
	releaseResources();
}

void Visualizer::releaseResources()
{
	if (screenTextureCuda_)
	{
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(screenTextureCuda_));
		screenTextureCuda_ = nullptr;
	}
	if (screenTextureGL_)
	{
		glDeleteTextures(1, &screenTextureGL_);
		screenTextureGL_ = 0;
	}
	if (screenTextureCudaBuffer_)
	{
		CUMAT_SAFE_CALL(cudaFree(screenTextureCudaBuffer_));
		screenTextureCudaBuffer_ = nullptr;
	}
	if (postOutput_)
	{
		CUMAT_SAFE_CALL(cudaFree(postOutput_));
		postOutput_ = nullptr;
	}
}

void Visualizer::settingsSave()
{
	// save file dialog
	auto fileNameStr = pfd::save_file(
		"Save settings",
		settingsDirectory_,
		{ "Json file", "*.json" },
		true
	).result();
	if (fileNameStr.empty())
		return;

	auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
	fileNamePath = fileNamePath.replace_extension(".json");
	std::cout << "Save settings to " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();

	// Build json
	nlohmann::json settings;
	settings["version"] = 1;
	//camera
	settings["camera"] = cameraGui_.toJson();
	//computation mode
	settings["computationMode"] = computationMode_;
	settings["renderMode"] = renderMode_;
	//TF editor
	settings["tfEditor"] = {
		{"editor", editor_.toJson()},
		{"minDensity", minDensity_},
		{"maxDensity", maxDensity_},
		{"opacityScaling", opacityScaling_},
		{"showColorControlPoints", showColorControlPoints_},
		{"dvrUseShading", dvrUseShading_}
	};
	//render parameters
	settings["renderer"] = {
		{"isovalue", rendererArgs_.isovalue},
		{"stepsize", rendererArgs_.stepsize},
		{"filterMode", rendererArgs_.volumeFilterMode},
		{"binarySearchSteps", rendererArgs_.binarySearchSteps},
		{"aoSamples", rendererArgs_.aoSamples},
		{"aoRadius", rendererArgs_.aoRadius}
	};
	//shading
	settings["shading"] = {
		{"materialColor", materialColor},
		{"ambientLight", ambientLightColor},
		{"diffuseLight", diffuseLightColor},
		{"specularLight", specularLightColor},
		{"specularExponent", specularExponent},
		{"aoStrength", aoStrength},
		{"lightDirection", lightDirectionScreen},
		{"channel", channelMode_},
		{"flowWithInpainting", flowWithInpainting_},
		{"temporalSmoothing", temporalPostSmoothingPercentage_}
	};
	//dataset
	settings["dataset"] = {
		{"file", volumeFullFilename_},
		{"mipmap", volumeMipmapLevel_},
		{"filterMode", volumeMipmapFilterMode_}
	};
	//networks
	auto getNetworkNames = [](const auto& nx)
	{
		auto a = nlohmann::json::array();
		for (const auto& n : nx)
		{
			if (!n->filename().empty())
				a.push_back(n->filename());
		}
		return a;
	};
	settings["networks"] = {
		{"sr-iso", getNetworkNames(networksIso_)},
		{"sr-iso-selection", selectedNetworkIso_},
		{"sr-dvr", getNetworkNames(networksDvr_)},
		{"sr-dvr-selection", selectedNetworkDvr_},
		{"importance-iso", getNetworkNames(importanceSamplerIso_)},
		{"importance-iso-selection", selectedImportanceSamplerIso_},
		{"importance-dvr", getNetworkNames(importanceSamplerDvr_)},
		{"importance-dvr-selection", selectedImportanceSamplerDvr_},
		{"recon-iso", getNetworkNames(sparseNetworksIso_)},
		{"recon-iso-selection", selectedSparseNetworkIso_},
		{"recon-dvr", getNetworkNames(sparseNetworksDvr_)},
		{"recon-dvr-selection", selectedSparseNetworkDvr_},
		{"temporalConsistency", temporalConsistency_},
		{"superresUpscaleFactor", superresUpscaleFactor_},
		{"importanceUpscale", importanceUpscale_},
		{"samplingSequence", samplingSequenceFullFilename_},
		{"samplingMinImportance", samplingMinImportance_},
		{"samplingMeanImportance", samplingMeanImportance_},
		{"stepsize-importance", getNetworkNames(stepsizeImportanceSamplerDvr_)},
		{"stepsize-importance-selection", selectedStepsizeImportanceSamplerDvr_},
		{"stepsize-reconstruction", getNetworkNames(stepsizeReconstructionDvr_)},
		{"stepsize-reconstruction-selection", selectedStepsizeReconstructionDvr_}
	};

	//save json to file
	std::ofstream o(fileNamePath);
	o << std::setw(4) << settings << std::endl;
	screenshotString_ = std::string("Settings saved to ") + fileNamePath.string();
	screenshotTimer_ = 2.0f;
}

namespace
{
	std::string getDir(const std::string& path)
	{
		if (path.empty())
			return path;
		std::experimental::filesystem::path p(path);
		if (std::experimental::filesystem::is_directory(p))
			return path;
		return p.parent_path().string();
	}
}

void Visualizer::settingsLoad()
{
	// load file dialog
	auto results = pfd::open_file(
        "Load settings",
        getDir(settingsDirectory_),
        { "Json file", "*.json" },
        false
    ).result();
	if (results.empty())
		return;

	auto fileNameStr = results[0];
	auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
	std::cout << "Load settings from " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();
	const auto basePath = fileNamePath.parent_path();

	//load json
	std::ifstream i(fileNamePath);
	nlohmann::json settings;
	try
	{
		i >> settings;
	} catch (const nlohmann::json::exception& ex)
	{
		pfd::message("Unable to parse Json", std::string(ex.what()),
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}
	i.close();
	int version = settings.contains("version")
		? settings.at("version").get<int>()
		: 0;
	if (version != 1)
	{
		pfd::message("Illegal Json", "The loaded json does not contain settings in the correct format",
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}

	//Ask which part should be loaded
	static bool loadCamera, loadComputationMode, loadTFeditor, loadRenderer, loadShading;
	static bool loadDataset, loadNetworks;
	static bool popupOpened;
	loadCamera = settingsToLoad_ & CAMERA;
	loadComputationMode = settingsToLoad_ & COMPUTATION_MODE;
	loadTFeditor = settingsToLoad_ & TF_EDITOR;
	loadRenderer = settingsToLoad_ & RENDERER;
	loadShading = settingsToLoad_ & SHADING;
	loadDataset = settingsToLoad_ & DATASET;
	loadNetworks = settingsToLoad_ & NETWORKS;
	popupOpened = false;
	auto guiTask = [this, settings, basePath]()
	{
		if (!popupOpened)
		{
			ImGui::OpenPopup("What to load");
			popupOpened = true;
			std::cout << "Open popup" << std::endl;
		}
		if (ImGui::BeginPopupModal("What to load", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Checkbox("Camera##LoadSettings", &loadCamera);
			ImGui::Checkbox("Computation Mode##LoadSettings", &loadComputationMode);
			ImGui::Checkbox("TF Editor##LoadSettings", &loadTFeditor);
			ImGui::Checkbox("Renderer##LoadSettings", &loadRenderer);
			ImGui::Checkbox("Shading##LoadSettings", &loadShading);
			ImGui::Checkbox("Dataset##LoadSettings", &loadDataset);
			ImGui::Checkbox("Networks##LoadSettings", &loadNetworks);
			if (ImGui::Button("Load##LoadSettings") || ImGui::IsKeyPressedMap(ImGuiKey_Enter))
			{
				try
				{
					//apply new settings
					if (loadCamera)
					{
						cameraGui_.fromJson(settings.at("camera"));
					}
					if (loadComputationMode)
					{
						computationMode_ = settings.at("computationMode").get<ComputationMode>();
						renderMode_ = settings.at("renderMode").get<RenderMode>();
					}
					if (loadTFeditor)
					{
						const auto& s = settings.at("tfEditor");
						editor_.fromJson(s.at("editor"));
						minDensity_ = s.at("minDensity").get<float>();
						maxDensity_ = s.at("maxDensity").get<float>();
						opacityScaling_ = s.at("opacityScaling").get<float>();
						showColorControlPoints_ = s.at("showColorControlPoints").get<bool>();
						dvrUseShading_ = s.at("dvrUseShading").get<bool>();
					}
					if (loadRenderer)
					{
						const auto& s = settings.at("renderer");
						rendererArgs_.isovalue = s.at("isovalue").get<double>();
						rendererArgs_.stepsize = s.at("stepsize").get<double>();
						rendererArgs_.volumeFilterMode = s.at("filterMode").get<renderer::RendererArgs::VolumeFilterMode>();
						rendererArgs_.binarySearchSteps = s.at("binarySearchSteps").get<int>();
						rendererArgs_.aoSamples = s.at("aoSamples").get<int>();
						rendererArgs_.aoRadius = s.at("aoRadius").get<double>();
					}
					if (loadShading)
					{
						const auto& s = settings.at("shading");
						materialColor = s.at("materialColor").get<float3>();
						ambientLightColor = s.at("ambientLight").get<float3>();
						diffuseLightColor = s.at("diffuseLight").get<float3>();
						specularLightColor = s.at("specularLight").get<float3>();
						specularExponent = s.at("specularExponent").get<float>();
						aoStrength = s.at("aoStrength").get<float>();
						lightDirectionScreen = s.at("lightDirection").get<float3>();
						channelMode_ = s.at("channel").get<ChannelMode>();
						flowWithInpainting_ = s.at("flowWithInpainting").get<bool>();
						temporalPostSmoothingPercentage_ = s.at("temporalSmoothing").get<int>();
					}
					if (loadDataset)
					{
						const auto& s = settings.at("dataset");
						if (!s.at("file").get<std::string>().empty())
						{
							auto targetPath = std::experimental::filesystem::path(s.at("file").get<std::string>());
							auto absPath = targetPath.is_absolute()
								? targetPath
								: std::experimental::filesystem::absolute(basePath / targetPath);
							try {
								loadVolume(absPath.string(), false);
								selectMipmapLevel(
									s.at("mipmap").get<int>(),
									s.at("filterMode").get<renderer::Volume::MipmapFilterMode>(),
									false);
							}
							catch (const std::exception& ex)
							{
								std::cerr << "Unable to load dataset with path " << absPath << ": " << ex.what() << std::endl;
							}
						}
					}
					if (loadNetworks)
					{
						const auto& s = settings.at("networks");
						auto loadNetwork = [basePath](auto& nx, const nlohmann::json::array_t& names, const auto& factory)
						{
							//remove all networks that can be deleted
							for (int i=nx.size()-1; i>=0; --i)
							{
								if (nx[i]->canDelete)
									nx.erase(nx.begin() + i);
							}
							//add networks
							for (const auto& entry : names)
							{
								const std::string name = entry.get<std::string>();
								auto targetPath = std::experimental::filesystem::path(name);
								auto absPath = targetPath.is_absolute()
									? targetPath
									: std::experimental::filesystem::absolute(basePath / targetPath);
								try
								{
									auto net = factory(absPath.string());
									nx.push_back(net);
								}
								catch (const std::exception& ex)
								{
									std::cerr << "unable to load network " << absPath << ": " << ex.what();
								}
							}
						};

						loadNetwork(networksIso_, s.at("sr-iso"), [](const std::string& name) {return std::make_shared<LoadedSuperresolutionModelIso>(name); });
						selectedNetworkIso_ = s.at("sr-iso-selection").get<int>();
						loadNetwork(networksDvr_, s.at("sr-dvr"), [](const std::string& name) {return std::make_shared<LoadedSuperresolutionModelDvr>(name); });
						selectedNetworkDvr_ = s.at("sr-dvr-selection").get<int>();

						loadNetwork(importanceSamplerIso_, s.at("importance-iso"), [](const std::string& name) {return std::make_shared<ImportanceSamplingNetwork>(name, RenderMode::IsosurfaceRendering); });
						selectedImportanceSamplerIso_ = s.at("importance-iso-selection").get<int>();
						loadNetwork(importanceSamplerDvr_, s.at("importance-dvr"), [](const std::string& name) {return std::make_shared<ImportanceSamplingNetwork>(name, RenderMode::DirectVolumeRendering); });
						selectedImportanceSamplerDvr_ = s.at("importance-dvr-selection").get<int>();

						loadNetwork(sparseNetworksIso_, s.at("recon-iso"), [](const std::string& name) {return std::make_shared<AdaptiveReconstructionModel>(name, RenderMode::IsosurfaceRendering); });
						selectedSparseNetworkIso_ = s.at("recon-iso-selection").get<int>();
						loadNetwork(sparseNetworksDvr_, s.at("recon-dvr"), [](const std::string& name) {return std::make_shared<AdaptiveReconstructionModel>(name, RenderMode::DirectVolumeRendering); });
						selectedSparseNetworkDvr_ = s.at("recon-dvr-selection").get<int>();

						temporalConsistency_ = s.at("temporalConsistency").get<bool>();
						superresUpscaleFactor_ = s.at("superresUpscaleFactor").get<int>();
						importanceUpscale_ = s.at("importanceUpscale").get<int>();
						samplingMinImportance_ = s.at("samplingMinImportance").get<float>();
						samplingMeanImportance_ = s.at("samplingMeanImportance").get<float>();

						if (!s.at("samplingSequence").get<std::string>().empty())
						{
							auto targetPath = std::experimental::filesystem::path(s.at("samplingSequence").get<std::string>());
							auto absPath = targetPath.is_absolute()
								? targetPath
								: std::experimental::filesystem::absolute(basePath / targetPath);
							try {
								loadSamplingSequence(absPath.string());
							} catch (const std::exception& ex)
							{
								std::cerr << "unable to load sampling pattern " << absPath << ": " << ex.what();
							}
						}

						loadNetwork(stepsizeImportanceSamplerDvr_, s.at("stepsize-importance"), [](const std::string& name) {return std::make_shared<ImportanceSamplingNetwork>(name, RenderMode::DirectVolumeRendering); });
						selectedStepsizeImportanceSamplerDvr_ = s.at("stepsize-importance-selection").get<int>();
						loadNetwork(stepsizeReconstructionDvr_, s.at("stepsize-reconstruction"), [](const std::string& name) {return std::make_shared<AdaptiveStepsizeReconstructionModel>(name, RenderMode::DirectVolumeRendering); });
						selectedStepsizeReconstructionDvr_ = s.at("stepsize-reconstruction-selection").get<int>();
					}
					//save last selection
					settingsToLoad_ =
						(loadCamera ? CAMERA : 0) |
						(loadComputationMode ? COMPUTATION_MODE : 0) |
						(loadTFeditor ? TF_EDITOR : 0) |
						(loadRenderer ? RENDERER : 0) |
						(loadShading ? SHADING : 0) |
						(loadDataset ? DATASET : 0) |
						(loadNetworks ? NETWORKS : 0);
					ImGui::MarkIniSettingsDirty();
					ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
					std::cout << "Settings applied" << std::endl;
				} catch (const nlohmann::json::exception& ex)
				{
					std::cerr << "Error: id=" << ex.id << ", message: " << ex.what() << std::endl;
					pfd::message("Unable to apply settings",
						std::string(ex.what()),
						pfd::choice::ok, pfd::icon::error).result();
				}
				//close popup
				this->backgroundGui_ = {};
				ImGui::CloseCurrentPopup();
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel##LoadSettings") || ImGui::IsKeyPressedMap(ImGuiKey_Escape))
			{
				//close popup
				this->backgroundGui_ = {};
				ImGui::CloseCurrentPopup();
				triggerRedraw(RedrawRenderer);
			}
			ImGui::EndPopup();
		}
	};
	worker_.wait(); //wait for current task
	this->backgroundGui_ = guiTask;
}

void Visualizer::loadVolume(const std::string& filename, bool background)
{
	std::string fileNameStr;
	if (filename.empty())
	{
		std::cout << "Open file dialog" << std::endl;
		// open file dialog
		auto results = pfd::open_file(
			"Load volume",
			getDir(volumeDirectory_),
			{ "Volumes", "*.dat *.xyz *.cvol" },
			false
		).result();
		if (results.empty())
			return;
		fileNameStr = results[0];
	}
	else
		fileNameStr = filename;

	std::cout << "Load " << fileNameStr << std::endl;
	auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
	volumeDirectory_ = fileNamePath.string();
	ImGui::MarkIniSettingsDirty();
	ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);

	//load the file
	std::shared_ptr<float> progress = std::make_shared<float>(0);
	auto loaderTask = [fileNameStr, fileNamePath, progress, background, this](BackgroundWorker* worker)
	{
		//callbacks
		renderer::VolumeProgressCallback_t progressCallback = [progress](float v)
		{
			*progress.get() = v * 0.99f;
		};
		renderer::VolumeLoggingCallback_t logging = [](const std::string& msg)
		{
			std::cout << msg << std::endl;
		};
		int errorCode = 1;
		renderer::VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
		{
			errorCode = code;
			std::cerr << msg << std::endl;
		};
		//load it locally
		std::unique_ptr<renderer::Volume> volume;
		if (fileNamePath.extension() == ".dat")
			volume.reset(renderer::loadVolumeFromRaw(fileNameStr, progressCallback, logging, error));
		else if (fileNamePath.extension() == ".xyz")
			volume.reset(renderer::loadVolumeFromXYZ(fileNameStr, progressCallback, logging, error));
		else if (fileNamePath.extension() == ".cvol")
			volume = std::make_unique<renderer::Volume>(fileNameStr, progressCallback, logging, error);
		else {
			std::cerr << "Unrecognized extension: " << fileNamePath.extension() << std::endl;
		}
		if (volume != nullptr) {
			volume->getLevel(0)->copyCpuToGpu();
			std::swap(volume_, volume);
			volumeMipmapLevel_ = 0;
			volumeFilename_ = fileNamePath.filename().string();
			volumeFullFilename_ = fileNamePath.string();
			std::cout << "Loaded" << std::endl;

			volumeHistogram_ = volume_->extractHistogram();

			minDensity_ = (minDensity_ < volumeHistogram_.maxDensity && minDensity_ > volumeHistogram_.minDensity) ? minDensity_ : volumeHistogram_.minDensity;
			maxDensity_ = (maxDensity_ < volumeHistogram_.maxDensity && maxDensity_ > volumeHistogram_.minDensity) ? maxDensity_ : volumeHistogram_.maxDensity;
		}

		//set it in the GUI and close popup
		if (background)
		{
			this->backgroundGui_ = {};
			ImGui::CloseCurrentPopup();
		}
		triggerRedraw(RedrawRenderer);
	};
	if (background)
	{
		worker_.wait(); //wait for current task
		auto guiTask = [progress]()
		{
			std::cout << "Progress " << *progress.get() << std::endl;
			if (ImGui::BeginPopupModal("Load Volume", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
			{
				ImGui::ProgressBar(*progress.get(), ImVec2(200, 0));
				ImGui::EndPopup();
			}
		};
		this->backgroundGui_ = guiTask;
		ImGui::OpenPopup("Load Volume");
		//start background task
		worker_.launch(loaderTask);
	} else
	{
		BackgroundWorker dummy;
		loaderTask(&dummy);
	}
}

void Visualizer::loadNetwork(RenderMode mode)
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load super-resolution network",
		getDir(mode==IsosurfaceRendering ? networkDirectoryIso_ : networkDirectoryDvr_),
		{ "PyTorch Script", "*.pt" },
		true
	).result();
	
	bool settingsWritten = false;
	for (const auto& fileNameStr : results)
	{
		std::cout << "Load " << fileNameStr << std::endl;
		if (!settingsWritten)
		{
			if (mode == IsosurfaceRendering)
				networkDirectoryIso_ = fileNameStr;
			else //DirectVolumeRendering
				networkDirectoryDvr_ = fileNameStr;
			ImGui::MarkIniSettingsDirty();
			ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
			settingsWritten = true;
		}
		try
		{
			if (mode == IsosurfaceRendering)
				networksIso_.push_back(std::make_shared<LoadedSuperresolutionModelIso>(fileNameStr));
			else //DirectVolumeRendering
				networksDvr_.push_back(std::make_shared<LoadedSuperresolutionModelDvr>(fileNameStr));
		}
		catch (const std::exception& ex)
		{
			std::cerr << "unable to load network: " << ex.what();
		}
	}
}

void Visualizer::loadSamplingPattern()
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load sampling pattern / mask",
		getDir(samplingPatternDirectory_),
		{ "Sampling Pattern", "*.txt" },
		false
	).result();
	if (results.empty())
		return;

	std::string fileNameStr = results[0];

	std::cout << "Load " << fileNameStr << std::endl;
	auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
	samplingPatternDirectory_ = fileNamePath.string();
	ImGui::MarkIniSettingsDirty();
	ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);

	try
	{
		auto[samples, triangles, width, height] =
			renderer::loadSamplesFromFile(fileNameStr);
		samplingPatternSize_ = make_int2(width, height);
		samplingPattern_ = samples.to(at::kCUDA);
		samplingPatternFilename_ = fileNamePath.filename().string();
		//create mesh drawer
		GLuint numVertices = samplingPattern_.size(1);
		samplingMeshDrawer_ = std::make_unique<renderer::MeshDrawer>(numVertices, triangles);
	}
	catch (const std::exception& ex)
	{
		std::cerr << "unable to load sampling pattern: " << ex.what();
	}
}

void Visualizer::loadSparseNetwork(RenderMode renderMode)
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load sparse reconstruction network",
		getDir((renderMode==IsosurfaceRendering) ? sparseNetworkDirectoryIso_ : sparseNetworkDirectoryDvr_),
		{ "PyTorch Script", "*.pt" },
		true
	).result();
	
	bool settingsWritten = false;
	for (const auto& fileNameStr : results)
	{
		std::cout << "Load " << fileNameStr << std::endl;
		if (!settingsWritten)
		{
			if (renderMode == IsosurfaceRendering)
				sparseNetworkDirectoryIso_ = fileNameStr;
			else
				sparseNetworkDirectoryDvr_ = fileNameStr;
			ImGui::MarkIniSettingsDirty();
			ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
			settingsWritten = true;
		}
		try {
			auto network = std::make_shared<AdaptiveReconstructionModel>(fileNameStr, renderMode);
			if (renderMode == IsosurfaceRendering)
				sparseNetworksIso_.push_back(network);
			else
				sparseNetworksDvr_.push_back(network);
		}
		catch (const std::exception& ex)
		{
			std::cerr << "unable to load network: " << ex.what();
		}
	}
}

void Visualizer::loadSamplingSequence(const std::string& filename)
{
	std::string fileNameStr;
	if (filename.empty())
	{
		std::cout << "Open file dialog" << std::endl;
		auto results = pfd::open_file(
			"Load sampling sequence",
			getDir(samplingSequenceDirectory_),
			{ "Sampling Sequences", "*.bin" },
			false
		).result();

		if (results.empty())
			return;

		fileNameStr = results[0];
	}
	else
		fileNameStr = filename;

	std::cout << "Load " << fileNameStr << std::endl;
	auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
	samplingSequenceDirectory_ = fileNamePath.string();
	ImGui::MarkIniSettingsDirty();
	ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);

	try
	{
		samplingSequenceFullFilename_ = fileNamePath.string();
		samplingSequenceFilename_ = fileNamePath.filename().string();
		std::ifstream in(samplingSequenceFilename_, std::ifstream::binary);
		int width, height;
		in.read(reinterpret_cast<char*>(&width), sizeof(int));
		in.read(reinterpret_cast<char*>(&height), sizeof(int));
		samplingSequence_ = torch::empty(
			{ width, height }, at::dtype(c10::ScalarType::Float));
		in.read(reinterpret_cast<char*>(samplingSequence_.data_ptr()), sizeof(float)*width*height);
		samplingSequence_ = samplingSequence_.to(c10::kCUDA);
		std::cout << "Sampling sequence loaded: size=(" << samplingSequence_.size(1) <<
			", " << samplingSequence_.size(0) << "); min=" <<
			torch::min(samplingSequence_).item().toFloat() << ", max=" <<
			torch::max(samplingSequence_).item().toFloat() << std::endl;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "unable to load sampling sequence: " << ex.what();
	}
}

void Visualizer::loadImportanceNetwork(RenderMode renderMode, std::string& directory, std::vector<ImportanceSamplingMethodPtr>& networks)
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load importance sample network",
		getDir(directory),
		{ "PyTorch Script", "*.pt" },
		true
	).result();
	
	bool settingsWritten = false;
	for (const auto& fileNameStr : results)
	{
		std::cout << "Load " << fileNameStr << std::endl;
		if (!settingsWritten)
		{
			directory = getDir(fileNameStr);
			ImGui::MarkIniSettingsDirty();
			ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
			settingsWritten = true;
		}
		try {
			auto network = std::make_shared<ImportanceSamplingNetwork>(fileNameStr, renderMode);
			networks.push_back(network);
		}
		catch (const std::exception& ex)
		{
			std::cerr << "unable to load network: " << ex.what();
		}
	}
}

void Visualizer::loadStepsizeReconstructionNetwork(RenderMode renderMode)
{
	assert(renderMode == DirectVolumeRendering);
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load sparse reconstruction network",
		getDir(stepsizeReconstructionDirectoryDvr_),
		{ "PyTorch Script", "*.pt" },
		true
	).result();

	bool settingsWritten = false;
	for (const auto& fileNameStr : results)
	{
		std::cout << "Load " << fileNameStr << std::endl;
		if (!settingsWritten)
		{
			stepsizeReconstructionDirectoryDvr_ = fileNameStr;
			ImGui::MarkIniSettingsDirty();
			ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
			settingsWritten = true;
		}
		try {
			auto network = std::make_shared<AdaptiveStepsizeReconstructionModel>(fileNameStr, renderMode);
			stepsizeReconstructionDvr_.push_back(network);
		}
		catch (const std::exception& ex)
		{
			std::cerr << "unable to load network: " << ex.what();
		}
	}
}

static void HelpMarker(const char* desc)
{
	//ImGui::TextDisabled(ICON_FA_QUESTION);
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}
void Visualizer::specifyUI()
{
	uiMenuBar();

	ImGui::PushItemWidth(ImGui::GetFontSize() * -8);

	uiVolume();
	uiCamera();
	uiTfEditor();
	uiRenderer();
	uiComputationMode();
	switch (renderMode_)
	{
	case IsosurfaceRendering:
		switch (computationMode_)
		{
		case ComputeSuperresolution:
			uiSuperResolution(IsosurfaceRendering);
			uiFoveated();
			break;
		case ComputeAdaptiveSampling:
			uiAdaptiveSampling(IsosurfaceRendering);
			break;
		case ComputeAdaptiveStepsize:
			uiAdaptiveStepsize(IsosurfaceRendering);
			break;
		}
		break;
	case DirectVolumeRendering:
		switch (computationMode_)
		{
		case ComputeSuperresolution:
			uiSuperResolution(DirectVolumeRendering);
			break;
		case ComputeAdaptiveSampling:
			uiAdaptiveSampling(DirectVolumeRendering);
			break;
		case ComputeAdaptiveStepsize:
			uiAdaptiveStepsize(DirectVolumeRendering);
			break;
		}
	}
	uiFoveatedMouse();
	uiShading();

	ImGui::PopItemWidth();

	if (backgroundGui_)
		backgroundGui_();

	uiScreenshotOverlay();
	uiFPSOverlay();
	uiLockMouseOverlay();
}

void Visualizer::uiMenuBar()
{
	ImGui::BeginMenuBar();
	ImGui::Text("Hotkeys");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted("'P': Screenshot");
		ImGui::TextUnformatted("'L': Lock foveated center");
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
	if (ImGui::SmallButton("Save##Settings"))
		settingsSave();
	if (ImGui::SmallButton("Load##Settings"))
		settingsLoad();
	ImGui::EndMenuBar();
	//hotkeys
	if (ImGui::IsKeyPressed(GLFW_KEY_P, false))
	{
		screenshot();
	}
	if (ImGui::IsKeyPressed(GLFW_KEY_L, false))
		foveatedLockMouse_ = !foveatedLockMouse_;
}

void Visualizer::selectMipmapLevel(int level, renderer::Volume::MipmapFilterMode filter, bool background)
{
	if (volume_ == nullptr) return;
	if (level == volumeMipmapLevel_ &&
		filter == volumeMipmapFilterMode_) return;
	if (volume_->getLevel(level) == nullptr || filter != volumeMipmapFilterMode_)
	{
		if (background)
		{
			//resample in background thread
			worker_.wait(); //wait for current task
			auto guiTask = []()
			{
				if (ImGui::BeginPopupModal("Resample", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
				{
					const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
					ImGuiExt::Spinner("ResampleVolume", 50, 10, col);
					ImGui::EndPopup();
				}
			};
			this->backgroundGui_ = guiTask;
			ImGui::OpenPopup("Resample");
			auto resampleTask = [level, filter, this](BackgroundWorker* worker)
			{
				if (filter != volumeMipmapFilterMode_)
					volume_->deleteAllMipmapLevels();
				volume_->createMipmapLevel(level, filter);
				volume_->getLevel(level)->copyCpuToGpu();
				volumeMipmapLevel_ = level;
				volumeMipmapFilterMode_ = filter;
				//close popup
				this->backgroundGui_ = {};
				ImGui::CloseCurrentPopup();
				triggerRedraw(RedrawRenderer);
			};
			//start background task
			worker_.launch(resampleTask);
		}
		else
		{
			if (filter != volumeMipmapFilterMode_)
				volume_->deleteAllMipmapLevels();
			volume_->createMipmapLevel(level, filter);
			volume_->getLevel(level)->copyCpuToGpu();
			volumeMipmapLevel_ = level;
			volumeMipmapFilterMode_ = filter;
			triggerRedraw(RedrawRenderer);
		}
	}
	else
	{
		//just ensure, it is on the GPU
		volume_->getLevel(level)->copyCpuToGpu();
		volumeMipmapLevel_ = level;
		volumeMipmapFilterMode_ = filter; //not necessarily needed
		triggerRedraw(RedrawRenderer);
	}
}

void Visualizer::uiVolume()
{
	if (ImGui::CollapsingHeader("Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::InputText("", &volumeFilename_[0], volumeFilename_.size() + 1, ImGuiInputTextFlags_ReadOnly);
		ImGui::SameLine();
		if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Volume"))
		{
			loadVolume();
		}

		//Level buttons
		for (int i = 0; i < sizeof(MipmapLevels) / sizeof(int); ++i)
		{
			int l = MipmapLevels[i];
			if (i > 0) ImGui::SameLine();
			std::string label = std::to_string(l + 1) + "x";
			if (ImGui::RadioButton(label.c_str(), volumeMipmapLevel_ == l))
				selectMipmapLevel(l, volumeMipmapFilterMode_);
		}
		//Filter buttons
		ImGui::TextUnformatted("Filtering:");
		ImGui::SameLine();
		if (ImGui::RadioButton("Average",
			volumeMipmapFilterMode_ == renderer::Volume::MipmapFilterMode::AVERAGE))
			selectMipmapLevel(volumeMipmapLevel_, renderer::Volume::MipmapFilterMode::AVERAGE);
		ImGui::SameLine();
		if (ImGui::RadioButton("Halton",
			volumeMipmapFilterMode_ == renderer::Volume::MipmapFilterMode::HALTON))
			selectMipmapLevel(volumeMipmapLevel_, renderer::Volume::MipmapFilterMode::HALTON);

		//print statistics
		ImGui::Text("Resolution: %d, %d, %d\nSize: %5.3f, %5.3f, %5.3f",
			volume_ && volume_->getLevel(volumeMipmapLevel_) ? static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeX()) : 0,
			volume_ && volume_->getLevel(volumeMipmapLevel_) ? static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeY()) : 0,
			volume_ && volume_->getLevel(volumeMipmapLevel_) ? static_cast<int>(volume_->getLevel(volumeMipmapLevel_)->sizeZ()) : 0,
			volume_ ? volume_->worldSizeX() : 0,
			volume_ ? volume_->worldSizeY() : 0,
			volume_ ? volume_->worldSizeZ() : 0);

		if (volume_)
		{
			ImGui::Text("Min Density: %f\nMax Density: %f", volumeHistogram_.minDensity, volumeHistogram_.maxDensity);
		}
	}
}

void Visualizer::uiCamera()
{
	if (ImGui::CollapsingHeader("Camera")) {
		if (cameraGui_.specifyUI()) triggerRedraw(RedrawRenderer);
	}
	if (cameraGui_.updateMouse())
		triggerRedraw(RedrawRenderer);
}

void Visualizer::uiRenderer()
{
	if (ImGui::CollapsingHeader("Render Parameters")) {
		if (renderMode_ == IsosurfaceRendering)
		{
			double isoMin = 0.01, isoMax = 2.0;
			if (ImGui::SliderScalar("Isovalue", ImGuiDataType_Double, &rendererArgs_.isovalue, &isoMin, &isoMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);
		}
		double stepMin = 0.01, stepMax = 1.0;
		if (ImGui::SliderScalar("Stepsize", ImGuiDataType_Double, &rendererArgs_.stepsize, &stepMin, &stepMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);
		static const char* VolumeFilterModeNames[] = { "Trilinear", "Tricubic" };
		const char* currentFilterModeName = (rendererArgs_.volumeFilterMode >= 0 && rendererArgs_.volumeFilterMode < renderer::RendererArgs::_VOLUME_FILTER_MODE_COUNT_)
			? VolumeFilterModeNames[rendererArgs_.volumeFilterMode] : "Unknown";
		if (ImGui::SliderInt("Filter Mode", reinterpret_cast<int*>(&rendererArgs_.volumeFilterMode),
			0, renderer::RendererArgs::_VOLUME_FILTER_MODE_COUNT_ - 1, currentFilterModeName))
			triggerRedraw(RedrawRenderer);
		if (renderMode_ == IsosurfaceRendering)
		{
			int binaryMin = 0, binaryMax = 10;
			if (ImGui::SliderScalar("Binary Search", ImGuiDataType_S32, &rendererArgs_.binarySearchSteps, &binaryMin, &binaryMax, "%d")) triggerRedraw(RedrawRenderer);
			int aoSamplesMin = 0, aoSamplesMax = 512;
			if (ImGui::SliderScalar("AO Samples", ImGuiDataType_S32, &rendererArgs_.aoSamples, &aoSamplesMin, &aoSamplesMax, "%d", 2)) triggerRedraw(RedrawRenderer);
			double aoRadiusMin = 0.01, aoRadiusMax = 0.5;
			if (ImGui::SliderScalar("AO Radius", ImGuiDataType_Double, &rendererArgs_.aoRadius, &aoRadiusMin, &aoRadiusMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);
		}
	}
}

void Visualizer::uiComputationMode()
{
	if (ImGui::RadioButton("Iso-surface",
		reinterpret_cast<int*>(&renderMode_), IsosurfaceRendering))
		triggerRedraw(RedrawRenderer);
	ImGui::SameLine();
	if (ImGui::RadioButton("Dvr",
		reinterpret_cast<int*>(&renderMode_), DirectVolumeRendering))
		triggerRedraw(RedrawRenderer);

	if (ImGui::RadioButton("Super-Res",
		reinterpret_cast<int*>(&computationMode_), ComputeSuperresolution))
		triggerRedraw(RedrawRenderer);
	ImGui::SameLine();
	if (ImGui::RadioButton("Adpt. Sampling",
		reinterpret_cast<int*>(&computationMode_), ComputeAdaptiveSampling))
		triggerRedraw(RedrawRenderer);
	ImGui::SameLine();
	if (ImGui::RadioButton("Adpt. Stepsize",
		reinterpret_cast<int*>(&computationMode_), ComputeAdaptiveStepsize))
		triggerRedraw(RedrawRenderer);

	if (ImGui::Checkbox("Evaluate networks in 16-bit precision", &VisualizerEvaluateNetworksInHalfPrecision))
		triggerRedraw(RedrawRenderer);
}

void Visualizer::uiTfEditor()
{
	if (ImGui::CollapsingHeader("TF Editor"))
	{
		if (ImGui::Button(ICON_FA_FOLDER_OPEN " Load TF"))
		{
			// open file dialog
			auto results = pfd::open_file(
				"Load transfer function",
				tfDirectory_,
				{ "Transfer Function", "*.tf" },
				false
			).result();
			if (results.empty())
				return;
			std::string fileNameStr = results[0];

			auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
			std::cout << "TF is loaded from " << fileNamePath << std::endl;
			tfDirectory_ = fileNamePath.string();

			editor_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			triggerRedraw(RedrawRenderer);
		}
		ImGui::SameLine();
		if (ImGui::Button(ICON_FA_SAVE " Save TF"))
		{
			// save file dialog
			auto fileNameStr = pfd::save_file(
				"Save transfer function",
				tfDirectory_,
				{ "Transfer Function", "*.tf" },
				true
			).result();
			if (fileNameStr.empty())
				return;

			auto fileNamePath = std::experimental::filesystem::path(fileNameStr);
			fileNamePath = fileNamePath.replace_extension(".tf");
			std::cout << "TF is saved under " << fileNamePath << std::endl;
			tfDirectory_ = fileNamePath.string();

			editor_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
		}
		ImGui::SameLine();
		ImGui::Checkbox("Show CPs", &showColorControlPoints_);

		ImGuiWindow* window = ImGui::GetCurrentWindow();
		ImGuiContext& g = *GImGui;
		const ImGuiStyle& style = g.Style;

		//Color
		const ImGuiID tfEditorColorId = window->GetID("TF Editor Color");
		auto pos = window->DC.CursorPos;
		auto tfEditorColorWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
		auto tfEditorColorHeight = 50.0f;
		const ImRect tfEditorColorRect(pos, ImVec2(pos.x + tfEditorColorWidth, pos.y + tfEditorColorHeight));
		ImGui::ItemSize(tfEditorColorRect, style.FramePadding.y);
		ImGui::ItemAdd(tfEditorColorRect, tfEditorColorId);

		//Opacity
		const ImGuiID tfEditorOpacityId = window->GetID("TF Editor Opacity");
		pos = window->DC.CursorPos;
		auto tfEditorOpacityWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
		auto tfEditorOpacityHeight = 100.0f;
		const ImRect tfEditorOpacityRect(pos, ImVec2(pos.x + tfEditorOpacityWidth, pos.y + tfEditorOpacityHeight));

		auto histogramRes = (volumeHistogram_.maxDensity - volumeHistogram_.minDensity) / volumeHistogram_.getNumOfBins();
		int histogramBeginOffset = (minDensity_ - volumeHistogram_.minDensity) / histogramRes;
		int histogramEndOffset = (volumeHistogram_.maxDensity - maxDensity_) / histogramRes;
		auto maxFractionVal = *std::max_element(std::begin(volumeHistogram_.bins) + histogramBeginOffset, std::end(volumeHistogram_.bins) - histogramEndOffset);
		ImGui::PlotHistogram("", volumeHistogram_.bins + histogramBeginOffset, volumeHistogram_.getNumOfBins() - histogramEndOffset - histogramBeginOffset,
			0, NULL, 0.0f, maxFractionVal, ImVec2(tfEditorOpacityWidth, tfEditorOpacityHeight));

		editor_.init(tfEditorOpacityRect, tfEditorColorRect, showColorControlPoints_);
		editor_.handleIO();
		editor_.render();

		if (ImGui::SliderFloat("Opacity Scaling", &opacityScaling_, 1.0f, 500.0f))
		{
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::SliderFloat("Min Density", &minDensity_, volumeHistogram_.minDensity, volumeHistogram_.maxDensity))
		{
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::SliderFloat("Max Density", &maxDensity_, volumeHistogram_.minDensity, volumeHistogram_.maxDensity))
		{
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::Checkbox("Use Shading", &dvrUseShading_))
		{
			triggerRedraw(RedrawRenderer);
		}

		if (editor_.getIsChanged())
		{
			triggerRedraw(RedrawRenderer);
		}
	}
}

void Visualizer::uiSuperResolution(RenderMode renderMode)
{
	//selection of the mode
	std::vector<SuperresolutionMethodPtr>& networks = (renderMode == IsosurfaceRendering)
		? networksIso_ : networksDvr_;
	int& selectedNetwork = (renderMode == IsosurfaceRendering) ? selectedNetworkIso_ : selectedNetworkDvr_;

	//UI
	if (ImGui::CollapsingHeader("Super-Resolution", ImGuiTreeNodeFlags_DefaultOpen))
	{
		std::vector<const char*> networkNames(networks.size());
		for (int i = 0; i < networks.size(); ++i)
			networkNames[i] = networks[i]->name().c_str();
		if (ImGui::ListBox("", &selectedNetwork, networkNames.data(), networks.size()))
			triggerRedraw(RedrawRenderer);
		ImGui::SameLine();
		ImGui::BeginGroup();
		if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Network"))
		{
			loadNetwork(renderMode);
			for (auto& d : foveatedLayers_) d.possibleMethods_.clear();
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::ButtonEx(ICON_FA_MINUS "##Network", ImVec2(0, 0),
			networks[selectedNetwork]->canDelete ? 0 : ImGuiButtonFlags_Disabled))
		{
			networks.erase(networks.begin() + selectedNetwork);
			selectedNetwork--; //this assumes that the non-deletable are up front
			for (auto& d : foveatedLayers_) d.possibleMethods_.clear();
			triggerRedraw(RedrawRenderer);
		}
		ImGui::EndGroup();
		if (networks[selectedNetwork]->supportsArbitraryUpscaleFactors())
		{
			if (ImGui::SliderInt("Upscale Factor", &superresUpscaleFactor_, 1, 16))
				triggerRedraw(RedrawRenderer);
		}
		if (ImGui::Checkbox("Temporal Consistency", &temporalConsistency_))
			triggerRedraw(RedrawNetwork);
	}
}

void Visualizer::uiAdaptiveSampling(RenderMode renderMode)
{
	std::vector<ImportanceSamplingMethodPtr>& importanceSampler = (renderMode == IsosurfaceRendering)
		? importanceSamplerIso_ : importanceSamplerDvr_;
	int& selectedImportanceSampler = (renderMode == IsosurfaceRendering)
		? selectedImportanceSamplerIso_ : selectedImportanceSamplerDvr_;
	std::vector<AdaptiveReconstructionMethodPtr>& sparseNetworks = (renderMode == IsosurfaceRendering)
		? sparseNetworksIso_ : sparseNetworksDvr_;
	int& selectedSparseNetwork = (renderMode == IsosurfaceRendering)
		? selectedSparseNetworkIso_ : selectedSparseNetworkDvr_;
	
	if (ImGui::CollapsingHeader("Adaptive Sampling", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::TextUnformatted("Strategy:");
		ImGui::SameLine();
		if (ImGui::RadioButton("Dynamic##SparseNetworks",
			samplingStrategy_ == SamplingStrategyDynamic))
		{
			samplingStrategy_ = SamplingStrategyDynamic;
			triggerRedraw(RedrawRenderer);
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Fixed##SparseNetworks",
			samplingStrategy_ == SamplingStrategyFixed))
		{
			samplingStrategy_ = SamplingStrategyFixed;
			triggerRedraw(RedrawRenderer);
		}

		if (samplingStrategy_ == SamplingStrategyFixed)
		{
			ImGui::TextUnformatted("Sampling Pattern:");
			ImGui::InputText("##SamplingFilename", &samplingPatternFilename_[0], samplingPatternFilename_.size() + 1, ImGuiInputTextFlags_ReadOnly);
			ImGui::SameLine();
			if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Sampling"))
			{
				loadSamplingPattern();
				triggerRedraw(RedrawRenderer);
			}
			int N = samplingPattern_.defined() ? samplingPattern_.size(1) : 0;
			ImGui::Text("Num Samples: %d, size: %d x %d", N, samplingPatternSize_.x, samplingPatternSize_.y);
			if (ImGui::SliderFloat("Mask Scaling", &samplingZoom_, 1, 1.9))
				triggerRedraw(RedrawRenderer);

			ImGui::TextUnformatted("Interpolation:");
			ImGui::SameLine();
			if (ImGui::RadioButton("Barycentric##SparseNetworks",
				samplingInterpolationMode_ == SamplingInterpolationBarycentric))
			{
				samplingInterpolationMode_ = SamplingInterpolationBarycentric;
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("Inpainting##SparseNetworks",
				samplingInterpolationMode_ == SamplingInterpolaionInpainting))
			{
				samplingInterpolationMode_ = SamplingInterpolaionInpainting;
				triggerRedraw(RedrawRenderer);
			}
		}
		else // samplingStrategy_ == SamplingStrategyDynamic
		{
			ImGui::TextUnformatted("Importance Sampler:");
			std::vector<const char*> networkNames(importanceSampler.size());
			for (int i = 0; i < importanceSampler.size(); ++i)
				networkNames[i] = importanceSampler[i]->name().c_str();
			if (ImGui::ListBox("##ImportanceSampler",
				&selectedImportanceSampler,
				networkNames.data(), importanceSampler.size()))
				triggerRedraw(RedrawRenderer);
			ImGui::SameLine();
			ImGui::BeginGroup();
			if (ImGui::Button(ICON_FA_FOLDER_OPEN "##ImportanceSampler"))
			{
				loadImportanceNetwork(
					renderMode,
					(renderMode == IsosurfaceRendering) ? importanceNetDirectoryIso_ : importanceNetDirectoryDvr_,
					importanceSampler);
				triggerRedraw(RedrawRenderer);
			}
			if (ImGui::ButtonEx(ICON_FA_MINUS "##ImportanceSampler", ImVec2(0, 0),
				importanceSampler[selectedImportanceSampler]->canDelete ? 0 : ImGuiButtonFlags_Disabled))
			{
				importanceSampler.erase(importanceSampler.begin() + selectedImportanceSampler);
				selectedImportanceSampler--; //this assumes that the non-deletable are up front
				triggerRedraw(RedrawRenderer);
			}
			ImGui::EndGroup();
			std::string infoText = importanceSampler[selectedImportanceSampler]->infoString();
			if (!infoText.empty())
				ImGui::TextUnformatted(infoText.c_str());

			if (ImGui::SliderInt(
				"Upscaling##ImportanceSampler", &importanceUpscale_,
				1, 16))
				triggerRedraw(RedrawRenderer);

			ImGui::TextUnformatted("Test:");
			ImGui::SameLine();
			if (ImGui::Checkbox("Show Input##ImportanceSampler", &importanceTestShowInput_))
			{
				if (importanceTestShowInput_)
					importanceTestShowImportance_ = false;
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			if (ImGui::Checkbox("Show Importance##ImportanceSampler", &importanceTestShowImportance_))
			{
				if (importanceTestShowImportance_)
					importanceTestShowInput_ = false;
				triggerRedraw(RedrawRenderer);
			}

			ImGui::TextUnformatted("Sampling Sequence:");
			ImGui::InputText("##SamplingSequence",
				&samplingSequenceFilename_[0],
				samplingSequenceFilename_.size() + 1, ImGuiInputTextFlags_ReadOnly);
			ImGui::SameLine();
			if (ImGui::Button(ICON_FA_FOLDER_OPEN "##SamplingSequence"))
			{
				loadSamplingSequence();
				triggerRedraw(RedrawRenderer);
			}

			if (ImGui::SliderFloat("Min Importance", &samplingMinImportance_, 0.0f, 1.0f))
				triggerRedraw(RedrawRenderer);
			if (ImGui::SliderFloat("Mean Importance", &samplingMeanImportance_, 0.0f, 1.0f))
				triggerRedraw(RedrawRenderer);
		}

		ImGui::TextUnformatted("Reconstruction Network:");
		std::vector<const char*> networkNames(sparseNetworks.size());
		for (int i = 0; i < sparseNetworks.size(); ++i)
			networkNames[i] = sparseNetworks[i]->name().c_str();
		if (ImGui::ListBox("##SparseNetworks", &selectedSparseNetwork, networkNames.data(), sparseNetworks.size()))
			triggerRedraw(RedrawNetwork);
		ImGui::SameLine();
		ImGui::BeginGroup();
		if (ImGui::Button(ICON_FA_FOLDER_OPEN "##SparseNetwork"))
		{
			loadSparseNetwork(renderMode);
			triggerRedraw(RedrawNetwork);
		}
		if (ImGui::ButtonEx(ICON_FA_MINUS "##SparseNetwork", ImVec2(0, 0),
			sparseNetworks[selectedSparseNetwork]->canDelete ? 0 : ImGuiButtonFlags_Disabled))
		{
			sparseNetworks.erase(sparseNetworks.begin() + selectedSparseNetwork);
			selectedSparseNetwork--; //this assumes that the non-deletable are up front
			triggerRedraw(RedrawNetwork);
		}
		ImGui::EndGroup();
		std::string infoText = sparseNetworks[selectedSparseNetwork]->infoString();
		if (!infoText.empty())
			ImGui::TextUnformatted(infoText.c_str());
		if (ImGui::Checkbox("Temporal Consistency", &temporalConsistency_))
			triggerRedraw(RedrawNetwork);
		if (temporalConsistency_)
			if (ImGui::Checkbox("Force Flow to Zero", &samplingForceFlowToZero_))
				triggerRedraw(RedrawNetwork);

		if (!sparseNetworks[selectedSparseNetwork]->isGroundTruth())
		{
			extraFrameInformation_ << "\n# adaptive samples: " << samplingNumberOfSamplesTaken_
				<< " (" << std::setprecision(5)
				<< (samplingNumberOfSamplesTaken_*100.0f / displayWidth_ / displayHeight_)
				<< "%)";
		}
		//times in milliseconds
		float timeRenderingLow = std::accumulate(
			timeRenderingLow_.begin(), timeRenderingLow_.end(), 0.0f) /
			timeRenderingLow_.size() * 1000.0f;
		float timeRenderingSamples = std::accumulate(
			timeRenderingSamples_.begin(), timeRenderingSamples_.end(), 0.0f) /
			timeRenderingSamples_.size() * 1000.0f;
		float timeImportance = std::accumulate(
			timeImportance_.begin(), timeImportance_.end(), 0.0f) /
			timeImportance_.size() * 1000.0f;
		float timeReconstruction = std::accumulate(
			timeReconstruction_.begin(), timeReconstruction_.end(), 0.0f) /
			timeReconstruction_.size() * 1000.0f;
		extraFrameInformation_ << "\nrendering time: " << std::fixed << std::setprecision(2)
			<< (timeRenderingLow + timeRenderingSamples) << "ms";
		extraFrameInformation_ << "\nimportance network time: " << std::fixed << std::setprecision(2)
			<< timeImportance << "ms";
		extraFrameInformation_ << "\nrecon. network time: " << std::fixed << std::setprecision(2)
			<< timeReconstruction << "ms";
	}
}

void Visualizer::uiAdaptiveStepsize(RenderMode renderMode)
{
	if (renderMode != DirectVolumeRendering)
	{
		ImGui::TextColored(ImVec4(1, 0, 0, 1),
			"Adaptive Stepsize only works with DVR");
		return;
	}
	
	std::vector<ImportanceSamplingMethodPtr>& importanceSampler = 
		stepsizeImportanceSamplerDvr_;
	int& selectedImportanceSampler = selectedStepsizeImportanceSamplerDvr_;
	std::vector<AdaptiveStepsizeReconstructionMethodPtr>& reconstructionNetworks =
		stepsizeReconstructionDvr_;
	int& selectedReconstructionNetwork = selectedStepsizeReconstructionDvr_;

	if (ImGui::CollapsingHeader("Adaptive Stepsize", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::TextUnformatted("Importance Sampler:");
		std::vector<const char*> networkNames(importanceSampler.size());
		for (int i = 0; i < importanceSampler.size(); ++i)
			networkNames[i] = importanceSampler[i]->name().c_str();
		if (ImGui::ListBox("##ImportanceSampler",
			&selectedImportanceSampler,
			networkNames.data(), importanceSampler.size()))
			triggerRedraw(RedrawRenderer);
		ImGui::SameLine();
		ImGui::BeginGroup();
		if (ImGui::Button(ICON_FA_FOLDER_OPEN "##ImportanceSampler"))
		{
			loadImportanceNetwork(renderMode, stepsizeImportanceDirectoryDvr_, stepsizeImportanceSamplerDvr_);
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::ButtonEx(ICON_FA_MINUS "##ImportanceSampler", ImVec2(0, 0),
			importanceSampler[selectedImportanceSampler]->canDelete ? 0 : ImGuiButtonFlags_Disabled))
		{
			importanceSampler.erase(importanceSampler.begin() + selectedImportanceSampler);
			selectedImportanceSampler--; //this assumes that the non-deletable are up front
			triggerRedraw(RedrawRenderer);
		}
		ImGui::EndGroup();
		std::string infoText = importanceSampler[selectedImportanceSampler]->infoString();
		if (!infoText.empty())
			ImGui::TextUnformatted(infoText.c_str());

		if (ImGui::SliderInt(
			"Upscaling##ImportanceSampler", &importanceUpscale_,
			1, 16))
			triggerRedraw(RedrawRenderer);

		ImGui::TextUnformatted("Test:");
		ImGui::SameLine();
		if (ImGui::Checkbox("Show Input##ImportanceSampler", &importanceTestShowInput_))
		{
			if (importanceTestShowInput_)
				importanceTestShowImportance_ = false;
			triggerRedraw(RedrawRenderer);
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("Show Importance##ImportanceSampler", &importanceTestShowImportance_))
		{
			if (importanceTestShowImportance_)
				importanceTestShowInput_ = false;
			triggerRedraw(RedrawRenderer);
		}

		
		if (ImGui::SliderFloat("Min Importance", &samplingMinImportance_, 0.0f, 1.0f))
			triggerRedraw(RedrawRenderer);
		if (ImGui::SliderFloat("Mean Importance", &samplingMeanImportance_, 0.0f, 1.0f))
			triggerRedraw(RedrawRenderer);

		
		ImGui::TextUnformatted("Reconstruction Network:");
		networkNames.resize(reconstructionNetworks.size());
		for (int i = 0; i < reconstructionNetworks.size(); ++i)
			networkNames[i] = reconstructionNetworks[i]->name().c_str();
		if (ImGui::ListBox("##SparseNetworks", &selectedReconstructionNetwork, networkNames.data(), reconstructionNetworks.size()))
			triggerRedraw(RedrawNetwork);
		ImGui::SameLine();
		ImGui::BeginGroup();
		if (ImGui::Button(ICON_FA_FOLDER_OPEN "##SparseNetwork"))
		{
			loadStepsizeReconstructionNetwork(renderMode);
			triggerRedraw(RedrawNetwork);
		}
		if (ImGui::ButtonEx(ICON_FA_MINUS "##SparseNetwork", ImVec2(0, 0),
			reconstructionNetworks[selectedReconstructionNetwork]->canDelete ? 0 : ImGuiButtonFlags_Disabled))
		{
			reconstructionNetworks.erase(reconstructionNetworks.begin() + selectedReconstructionNetwork);
			selectedReconstructionNetwork--; //this assumes that the non-deletable are up front
			triggerRedraw(RedrawNetwork);
		}
		ImGui::EndGroup();
		infoText = reconstructionNetworks[selectedReconstructionNetwork]->infoString();
		if (!infoText.empty())
			ImGui::TextUnformatted(infoText.c_str());
		if (ImGui::Checkbox("Temporal Consistency", &temporalConsistency_))
			triggerRedraw(RedrawNetwork);
		if (temporalConsistency_)
			if (ImGui::Checkbox("Force Flow to Zero", &samplingForceFlowToZero_))
				triggerRedraw(RedrawNetwork);

		extraFrameInformation_ << "\naverage samples per voxel: " << averageSamplesPerVoxel_;
		extraFrameInformation_ << "\nminimal stepsize: " << minAdaptiveStepsize_;
		extraFrameInformation_ << "\nmaximal stepsize: " << maxAdaptiveStepsize_;
	}
}

void Visualizer::uiFoveated()
{
	if (ImGui::CollapsingHeader("Foveated"))
	{
		const auto fillPossibleMethods = [this](FoveatedLayerDesc& d, int factor)
		{
			d.possibleMethods_.clear();
			if (factor == 1)
				d.possibleMethods_.push_back(BaselineSuperresolution::GetMethod(
					BaselineSuperresolution::InterpolationNearest, 1, 6));
			else
			{
				d.possibleMethods_.push_back(nullptr);
				d.possibleMethods_.push_back(BaselineSuperresolution::GetMethod(
					BaselineSuperresolution::InterpolationNearest, factor, 6));
				d.possibleMethods_.push_back(BaselineSuperresolution::GetMethod(
					BaselineSuperresolution::InterpolationLinear, factor, 6));
				d.possibleMethods_.push_back(BaselineSuperresolution::GetMethod(
					BaselineSuperresolution::InterpolationBicubic, factor, 6));
				for (const auto& m : networksIso_)
				{
					if (m->canDelete && m->upscaleFactor() == factor)
						d.possibleMethods_.push_back(m);
				}
			}
		};
		if (ImGui::Checkbox("Enable Foveated blending", &foveatedEnable_))
			triggerRedraw(RedrawRenderer);
		ImGui::Text("Number of Layers:"); ImGui::SameLine();
		float sz = ImGui::GetFrameHeight();
		if (ImGui::ArrowButtonEx(
			"##left", ImGuiDir_Left, ImVec2(sz, sz),
			foveatedLayers_.size() == 2 ? ImGuiButtonFlags_Disabled : 0))
		{
			foveatedLayers_.pop_back();
			if (!foveatedLayers_.empty())
				foveatedLayers_[foveatedLayers_.size() - 1].radius_ = 1.0f;
			triggerRedraw(RedrawFoveated);
		}
		ImGui::SameLine();
		if (ImGui::ArrowButton("##right", ImGuiDir_Right))
		{
			if (foveatedLayers_.size() >= 1) {
				foveatedLayers_[foveatedLayers_.size() - 1].radius_ = (
					100 + (foveatedLayers_.size() >= 2 ? foveatedLayers_[foveatedLayers_.size() - 2].radius_ : 0)) / 2;
			}
			FoveatedLayerDesc newLayer;
			newLayer.radius_ = 100;
			fillPossibleMethods(newLayer, foveatedLayers_.size() + 1);
			newLayer.method_ = newLayer.possibleMethods_[0];
			foveatedLayers_.push_back(newLayer);
			triggerRedraw(RedrawFoveated);
		}
		ImGui::SameLine();
		ImGui::Text("%d", int(foveatedLayers_.size()));

		if (ImGui::SliderInt("Blur Radius", &fovatedBlurRadiusPercent_, 0, 100, "%d%%"))
			triggerRedraw(RedrawFoveated);
		ImGui::SameLine();
		HelpMarker("Percentage towards the next layer\n100%: smoothing over the whole range between layers.\n0%: sharp boundary");
		if (ImGui::Combo("Blur Shape", reinterpret_cast<int*>(&foveatedBlurShape_), FoveatedBlurShapeNames, kernel::_FoveatedBlurShapeCount_))
			triggerRedraw(RedrawFoveated);

		ImGui::Columns(3, "foveatedTable");
		ImGui::Text("Layer"); ImGui::NextColumn();
		ImGui::Text("Network"); ImGui::NextColumn();
		ImGui::Text("Radius"); ImGui::SameLine();
		HelpMarker("Size of the spherical region as percentage of the screen diagonal"); ImGui::NextColumn();
		ImGui::Separator();
		for (int i = 0; i < foveatedLayers_.size(); ++i)
		{
			ImGui::PushID(i);
			auto& d = foveatedLayers_[i];
			ImGui::Text("%dx", (i + 1)); ImGui::NextColumn();
			if (i == 0)
				ImGui::Text("Ground Truth");
			else {
				if (d.possibleMethods_.empty())
					fillPossibleMethods(d, i + 1);
				if (ImGui::BeginCombo("##c", d.method_ == nullptr ? "Disable" : d.method_->name().c_str()))
				{
					for (const auto& m : d.possibleMethods_)
						if (ImGui::Selectable(m == nullptr ? "Disable" : m->name().c_str(), d.method_ == m))
						{
							d.method_ = m;
							triggerRedraw(RedrawFoveated);
							if (i == foveatedLayers_.size() - 1) //full-screen layer
								triggerRedraw(RedrawNetwork);
						}
					ImGui::EndCombo();
				}
			}
			ImGui::NextColumn();
			if (i < foveatedLayers_.size() - 1) {
				if (ImGui::SliderInt("##r", &d.radius_,
					i > 0 ? foveatedLayers_[i - 1].radius_ + 1 : 1,
					foveatedLayers_[i + 1].radius_ - 1,
					"%d%%"))
					triggerRedraw(RedrawFoveated);
			}
			else
				ImGui::Text("100%%");
			ImGui::NextColumn();
			ImGui::PopID();
		}
		ImGui::Columns(1);
		if (ImGui::Checkbox("Show Masks", &foveatedShowMasks_))
			triggerRedraw(RedrawRenderer);
	}

	//statistics
	if (computationMode_ == ComputeSuperresolution && foveatedEnable_)
		extraFrameInformation_ << "\nfoveated: % of samples: " <<
		std::setprecision(4) << (100 * foveatedComputeNumberOfSamples());
}

void Visualizer::uiFoveatedMouse()
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse && !io.MouseDown[0] && ImGui::IsMousePosValid() && !foveatedLockMouse_) {
		int2 oldFoveatedCenter = foveatedCenter_;
		foveatedCenter_ = make_int2(io.MousePos.x, io.MousePos.y);
		if (oldFoveatedCenter.x != foveatedCenter_.x ||
			oldFoveatedCenter.y != foveatedCenter_.y)
			triggerRedraw(RedrawFoveated);

	}
}

void Visualizer::uiShading()
{
	if (ImGui::CollapsingHeader("Output - Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
		auto redraw = renderMode_ == IsosurfaceRendering
			? RedrawPost
			: RedrawRenderer;

		ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel;
		if (ImGui::ColorEdit3("Material Color", &materialColor.x, colorFlags)) triggerRedraw(redraw);
		if (ImGui::ColorEdit3("Ambient Light", &ambientLightColor.x, colorFlags)) triggerRedraw(redraw);
		if (ImGui::ColorEdit3("Diffuse Light", &diffuseLightColor.x, colorFlags)) triggerRedraw(redraw);
		if (ImGui::ColorEdit3("Specular Light", &specularLightColor.x, colorFlags)) triggerRedraw(redraw);
		float minSpecular = 0, maxSpecular = 64;
		if (ImGui::SliderScalar("Spec. Exp.", ImGuiDataType_Float, &specularExponent, &minSpecular, &maxSpecular, "%.3f", 2)) triggerRedraw(redraw);
		float minAO = 0, maxAO = 1;
		if (ImGui::SliderScalar("AO Strength", ImGuiDataType_Float, &aoStrength, &minAO, &maxAO)) triggerRedraw(RedrawPost);
		if (ImGuiExt::DirectionPicker2D("Light direction", &lightDirectionScreen.x, ImGuiExtDirectionPickerFlags_InvertXY))
			triggerRedraw(redraw);
		const char* currentChannelName = (channelMode_ >= 0 && channelMode_ < _ChannelCount_)
			? ChannelModeNames[channelMode_] : "Unknown";
		if (ImGui::SliderInt("Channel", reinterpret_cast<int*>(&channelMode_), 0, _ChannelCount_ - 1, currentChannelName))
			triggerRedraw(RedrawPost);
		if (channelMode_ == ChannelFlow)
			if (ImGui::Checkbox("Flow with Inpainting", &flowWithInpainting_))
				triggerRedraw(RedrawPost);
		if (ImGui::SliderInt("Temporal Smoothing", &temporalPostSmoothingPercentage_, 0, 100, "%d%%"))
			triggerRedraw(RedrawRenderer);
	}
}

void Visualizer::uiScreenshotOverlay()
{
	if (screenshotTimer_ <= 0) return;

	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y - 10);
	ImVec2 window_pos_pivot = ImVec2(0.5f, 1.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	//ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
	ImGui::Begin("Example: Simple overlay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::TextUnformatted(screenshotString_.c_str());
	ImGui::End();
	//ImGui::PopStyleVar(ImGuiStyleVar_Alpha);

	screenshotTimer_ -= io.DeltaTime;
}

void Visualizer::uiFPSOverlay()
{
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x - 5, 5);
	ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowBgAlpha(0.5f);
	ImGui::Begin("FPSDisplay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::Text("FPS: %.1f", io.Framerate);
	std::string extraText = extraFrameInformation_.str();
	if (!extraText.empty())
	{
		extraText = extraText.substr(1); //strip initial '\n'
		ImGui::TextUnformatted(extraText.c_str());
	}
	extraFrameInformation_ = std::stringstream();
	ImGui::End();
}


void Visualizer::uiLockMouseOverlay()
{
	if (!foveatedLockMouse_) return;
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x - 5, io.DisplaySize.y - 5);
	ImVec2 window_pos_pivot = ImVec2(1.0f, 1.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowBgAlpha(0.5f);
	ImGui::Begin("LockMouse", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::Text("foveated center locked", io.Framerate);
	ImGui::End();
}

renderer::RendererArgs Visualizer::setupRendererArgs(
	RenderMode renderMode, int upscaleFactor)
{
	cameraGui_.updateRenderArgs(rendererArgs_);
	renderer::RendererArgs args = rendererArgs_;
	args.cameraResolutionX = displayWidth_ / upscaleFactor;
	args.cameraResolutionY = displayHeight_ / upscaleFactor;
	args.cameraViewport = make_int4(0, 0, -1, -1);
	args.mipmapLevel = volumeMipmapLevel_;
	args.renderMode = (renderMode==IsosurfaceRendering)
		? renderer::RendererArgs::ISO_UNSHADED : renderer::RendererArgs::DVR;
	args.densityAxisOpacity = editor_.getDensityAxisOpacity();
	args.opacityAxis = editor_.getOpacityAxis();
	args.densityAxisColor = editor_.getDensityAxisColor();
	args.colorAxis = editor_.getColorAxis();
	args.opacityScaling = opacityScaling_;
	args.minDensity = minDensity_;
	args.maxDensity = maxDensity_;

	renderer::ShadingSettings shading;
	shading.ambientLightColor = ambientLightColor;
	shading.diffuseLightColor = diffuseLightColor;
	shading.specularLightColor = specularLightColor;
	shading.specularExponent = specularExponent;
	shading.materialColor = materialColor;
	shading.aoStrength = aoStrength;
	shading.lightDirection = normalize(cameraGui_.screenToWorld(lightDirectionScreen));
	args.shading = shading;
	args.dvrUseShading = dvrUseShading_;

	return args;
}

void Visualizer::render(int display_w, int display_h)
{
	try {
		resize(display_w, display_h);
		at::GradMode::set_enabled(false);

		if (foveatedEnable_ && foveatedShowMasks_)
		{
			drawFoveatedMask(screenTextureCudaBuffer_);
			copyBufferToOpenGL();
			drawer_.drawQuad(screenTextureGL_);
			return;
		}

		if (volume_ == nullptr) return;
		if (volume_->getLevel(volumeMipmapLevel_) == nullptr) return;

		if (redrawMode_ == RedrawNone)
		{
			//just draw the precomputed texture
			drawer_.drawQuad(screenTextureGL_);
			return;
		}

		switch (computationMode_)
		{
		case ComputeSuperresolution:
			if (renderMode_ == IsosurfaceRendering)
			{
				renderSuperresolutionIso();
			}
			else if (renderMode_ == DirectVolumeRendering)
			{
				renderSuperresolutionDvr();
			}
			break;
		case ComputeAdaptiveSampling:
			renderAdaptiveSamples(renderMode_);
			break;
		case ComputeAdaptiveStepsize:
			renderAdaptiveStepsize(renderMode_);
			break;
		}
	} catch (const std::exception& ex)
	{
		std::cerr << "Uncaught exception: " << ex.what() << std::endl;
	}
}

void Visualizer::renderAdaptiveSamples(RenderMode renderMode)
{
	AdaptiveReconstructionMethodPtr selectedNetwork = (renderMode==IsosurfaceRendering)
		? sparseNetworksIso_[selectedSparseNetworkIso_] : sparseNetworksDvr_[selectedSparseNetworkDvr_];

	int result = RenderAdaptiveSamplesCanReconstruct;
	//if (!selectedNetwork->isGroundTruth())
	//{
		//for the ground truth reconstruction, we don't need a sampling input
	if (samplingStrategy_ == SamplingStrategyFixed)
		result = renderAdaptiveSamples_SampleFixed(renderMode, selectedNetwork->requiresAO());
	else //SamplingStrategyDynamic
	{
		ImportanceSamplingMethodPtr importanceSampler = (renderMode == IsosurfaceRendering)
			? importanceSamplerIso_[selectedImportanceSamplerIso_] : importanceSamplerDvr_[selectedImportanceSamplerDvr_];
		result = renderAdaptiveSamples_SampleDynamic(renderMode, importanceSampler, true);
		if (result == RenderAdaptiveSamplesNeedsSampling)
			result = renderAdaptiveSamples_RenderSamples(renderMode, selectedNetwork->requiresAO());
	}
	//}

	if (result == RenderAdaptiveSamplesCanReconstruct ||
		((result == RenderAdaptiveSamplesNoData) && selectedNetwork->isGroundTruth()))
	{
		renderAdaptiveSamples_Reconstruct(renderMode, selectedNetwork);
		drawer_.drawQuad(screenTextureGL_);
	}
	else if (result == RenderAdaptiveSamplesAlreadyFilled)
	{
		drawer_.drawQuad(screenTextureGL_);
	}

	redrawMode_ = RedrawNone;
}

int Visualizer::renderAdaptiveSamples_SampleDynamic(RenderMode renderMode, ImportanceSamplingMethodPtr importanceSampler, bool clampAtOne)
{
	if (!(redrawMode_ == RedrawRenderer))
	{
		if (importanceTestShowInput_ || importanceTestShowImportance_)
			return RenderAdaptiveSamplesAlreadyFilled;
		else
		{
			if (samplingNumberOfSamplesTaken_ > 0)
				return RenderAdaptiveSamplesCanReconstruct;
			else
				return RenderAdaptiveSamplesNoData;
		}
	}

	//to cope for non-divisible screen sizes
	const int lowWidth = displayWidth_ / importanceUpscale_;
	const int lowHeight = displayHeight_ / importanceUpscale_;
	const int highWidth = lowWidth * importanceUpscale_;
	const int highHeight = lowHeight * importanceUpscale_;

	//configuration between ISO and DVR
	const int numChannels = (renderMode == IsosurfaceRendering)
        ? renderer::IsoRendererOutputChannels : renderer::DvrRendererOutputChannels;
	const int maskChannel = (renderMode == IsosurfaceRendering) ? 0 : 3;
	const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;

	//1. Render low resolution input
	renderer::RendererArgs args = setupRendererArgs(renderMode);
	args.cameraResolutionX = lowWidth;
	args.cameraResolutionY = lowHeight;
	args.cameraViewport = make_int4(0, 0, lowWidth, lowHeight);
	args.aoSamples = 0;

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	at::Tensor lowResInput = torch::empty(
        { numChannels, args.cameraResolutionY, args.cameraResolutionX },
        torch::dtype(torch::kFloat).device(torch::kCUDA));

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	auto start = std::chrono::steady_clock::now();
	render_gpu(volume_.get(), &args, lowResInput, stream);
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	auto end = std::chrono::steady_clock::now();
	while (timeRenderingLow_.size() > MAX_TIME_ENTRIES) timeRenderingLow_.pop_front();
	timeRenderingLow_.push_back(std::chrono::duration<float>(end - start).count());

	const int padW = displayWidth_ - highWidth;
	const int padH = displayHeight_ - highHeight;
	
	//flow inpainting
	bool needFlow = true;
	if (needFlow)
	{
		const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;
		const auto& flow = lowResInput.unsqueeze(0).narrow(1, flowChannel, 2);;
		const auto& mask = lowResInput.unsqueeze(0).narrow(1, 0, 1);
		rendererInpaintedFlow_ = kernel::interpolateCubic(kernel::inpaintFlow(mask, flow),
			highWidth, highHeight);
		rendererInpaintedFlow_ = torch::constant_pad_nd(rendererInpaintedFlow_, { 0,padW,0,padH }, 0);
	}
	
	if (importanceTestShowInput_)
	{
		//draw low resolution input
		
		auto highResInput = kernel::interpolateNearest(
            lowResInput.unsqueeze(0), highWidth, highHeight);
		highResInput = torch::constant_pad_nd(highResInput, { 0,padW,0,padH }, 0);
		if (renderMode == IsosurfaceRendering)
			selectChannelIso(channelMode_, highResInput, screenTextureCudaBuffer_);
		else
			selectChannelDvr(channelMode_, highResInput, screenTextureCudaBuffer_);
		copyBufferToOpenGL();
		return RenderAdaptiveSamplesAlreadyFilled;
	}

	//2. compute importance map

	//warp previous output
	torch::Tensor previousImportanceOutput;
	if (importanceSampler->requiresPrevious() && temporalConsistency_ &&
        previousImportanceNetworkOutput_.sizes() == c10::IntArrayRef{ highHeight , highWidth })
	{
		if (samplingForceFlowToZero_)
		{
			//ignore flow from the network
			previousImportanceOutput = previousImportanceNetworkOutput_.unsqueeze(0);
		}
		else
		{
			//use interpolated flow from the current frame to warp
			auto lowResInput2 = lowResInput.unsqueeze(0);
			const auto mask = lowResInput2.narrow(1, maskChannel, 1);
			auto lowResFlow = kernel::inpaintFlow(
                mask, lowResInput2.narrow(1, flowChannel, 2));
			auto prevImpOutput = previousImportanceNetworkOutput_.unsqueeze(0).unsqueeze(0);
			previousImportanceOutput = kernel::warpUpscale(
                prevImpOutput,
                lowResFlow,
                importanceUpscale_)[0];
		}
	}

	//evalutate sampler
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	start = std::chrono::steady_clock::now();
	importanceMap_ = importanceSampler->eval(lowResInput, importanceUpscale_, previousImportanceOutput);
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	end = std::chrono::steady_clock::now();
	while (timeImportance_.size() > MAX_TIME_ENTRIES) timeImportance_.pop_front();
	timeImportance_.push_back(std::chrono::duration<float>(end - start).count());

	previousImportanceNetworkOutput_ = importanceMap_.clone();

	//3. normalize importance map
	normalizedImportanceMap_ = ImportanceSamplingMethod::normalize(
        importanceMap_,
        samplingMinImportance_, samplingMeanImportance_,
        4 * importanceUpscale_, clampAtOne);
	normalizedImportanceMap_ = torch::constant_pad_nd(
		normalizedImportanceMap_,
        { 0,padW,0,padH }, 0);

	if (importanceTestShowImportance_)
	{
		auto highRes = normalizedImportanceMap_.unsqueeze(0).unsqueeze(0);
		highRes.clamp_max_(1.0f);
		if (renderMode == IsosurfaceRendering)
			selectChannelIso(ChannelMask, highRes, screenTextureCudaBuffer_);
		else
		{
			//Note: we use 'Iso' here and not 'DVR' as one would expect.
			//highRes contains only one channel and we want to write that channel to the screen.
			//Iso's ChannelMask extracts the first channel, but in DVR mode,
			//the ChannelMask refers to the third channel, alpha -> out of memory error
			selectChannelIso(ChannelMask, highRes, screenTextureCudaBuffer_);
		}
		copyBufferToOpenGL();
		return RenderAdaptiveSamplesAlreadyFilled;
	}

	return RenderAdaptiveSamplesNeedsSampling;
}
int Visualizer::renderAdaptiveSamples_RenderSamples(RenderMode renderMode, bool requiresAO)
{
	//4. sample importance map
	if (!samplingSequence_.defined())
		return RenderAdaptiveSamplesNoData; //no sequence loaded
	if (samplingSequence_.size(0) < displayHeight_ ||
		samplingSequence_.size(1) < displayWidth_)
	{
		std::cout << "Sequence image is too small for the display!" << std::endl;
		return RenderAdaptiveSamplesNoData;
	}
	
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	const auto start = std::chrono::steady_clock::now();

	auto croppedSamplingSequence = samplingSequence_
		.narrow(0, 0, displayHeight_)
		.narrow(1, 0, displayWidth_);
	//This selects the samples based on the importance map
	auto samplingMask = normalizedImportanceMap_ > croppedSamplingSequence;

	//5. get sample positions as a linear tensor of shape (2xN)
	auto samplePositions = torch::nonzero(samplingMask.t_())
		.toType(c10::ScalarType::Float)
		.transpose(0, 1)
		.contiguous();
	samplingNumberOfSamplesTaken_ = samplePositions.size(1);
	if (samplingNumberOfSamplesTaken_ == 0)
		return RenderAdaptiveSamplesNoData;

	//6. render samples
	//configuration between ISO and DVR
	const int numChannels = (renderMode == IsosurfaceRendering)
		? renderer::IsoRendererOutputChannels : renderer::DvrRendererOutputChannels;
	const int maskChannel = (renderMode == IsosurfaceRendering) ? 0 : 3;
	const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;

	//create output
	samplingOutput_ = torch::empty(
		{ numChannels, samplingNumberOfSamplesTaken_ },
		torch::dtype(torch::kFloat).device(torch::kCUDA));

	//settings
	renderer::RendererArgs args = setupRendererArgs(renderMode);
	args.cameraResolutionX = displayWidth_;
	args.cameraResolutionY = displayHeight_;
	args.cameraViewport = make_int4(0, 0, displayWidth_, displayHeight_);
	args.aoSamples = requiresAO ? rendererArgs_.aoSamples : 0;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	//render samples
	render_samples_gpu(volume_.get(), &args, samplePositions, samplingOutput_, stream);

	//7. scatter them to the image
	rendererSparseOutput_ = torch::zeros(
		{ numChannels, displayHeight_, displayWidth_ },
		torch::dtype(torch::kFloat).device(torch::kCUDA));
	if (renderMode == IsosurfaceRendering)
		rendererSparseOutput_[maskChannel].fill_(0.5f);
	rendererSparseMask_ = torch::zeros(
		{ 1, displayHeight_, displayWidth_ },
		torch::dtype(torch::kFloat).device(torch::kCUDA));
	renderer::scatter_samples_to_image_gpu(
		samplePositions, samplingOutput_, rendererSparseOutput_, rendererSparseMask_, stream);
	if (renderMode == IsosurfaceRendering)
		rendererSparseOutput_[maskChannel] = rendererSparseOutput_[maskChannel] * 2 - 1;
	rendererSparseOutput_ = rendererSparseOutput_.unsqueeze(0);
	rendererSparseMask_ = rendererSparseMask_.unsqueeze(0);
	//and interpolate them
	//const auto mask = at::abs(rendererSparseOutput_.narrow(1, maskChannel, 1));
	rendererInterpolatedOutput_ = kernel::inpaintFlow(rendererSparseMask_, rendererSparseOutput_);

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	auto end = std::chrono::steady_clock::now();
	while (timeRenderingSamples_.size() > MAX_TIME_ENTRIES) timeRenderingSamples_.pop_front();
	timeRenderingSamples_.push_back(std::chrono::duration<float>(end - start).count());

	//done
	redrawMode_ = RedrawNetwork;
	return RenderAdaptiveSamplesCanReconstruct;
}

int Visualizer::renderAdaptiveSamples_SampleFixed(RenderMode renderMode, bool requiresAO)
{
	int64_t N = samplingPattern_.defined() ? samplingPattern_.size(1) : 0;
	if (N == 0)
		return RenderAdaptiveSamplesNoData;

	if ((redrawMode_ == RedrawRenderer || redrawMode_ == RedrawFoveated) && N > 0)
	{
		//rescale and move pattern to fit the screen size
		//this takes into account the samplingZoom_ and foveatedCenter_
		//samplingZoom_ = 1 -> pattern scaled to fit the whole screen, no movement
		//samplingZoom_ = 2 -> pattern twice the screen size, center at foveatedCenter_

		//configuration between ISO and DVR
		const int numChannels = (renderMode == IsosurfaceRendering)
			? renderer::IsoRendererOutputChannels : renderer::DvrRendererOutputChannels;
		const int maskChannel = (renderMode == IsosurfaceRendering) ? 0 : 3;
		const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;

		float factor = std::max(
			displayWidth_ * samplingZoom_ / float(samplingPatternSize_.x),
			displayHeight_ * samplingZoom_ / float(samplingPatternSize_.y));
		const glm::vec2 screenSize = glm::vec2(displayWidth_, displayHeight_);
		//centers the pattern in the center of the screen
		glm::vec2 baseOffset = -0.5f * (samplingZoom_ - 1) * screenSize;
		//offset of the foveated center to the screen center
		glm::vec2 foveatedOffset = glm::vec2(foveatedCenter_.x, foveatedCenter_.y) - 0.5f * screenSize;
		//allowed movement
		glm::vec2 allowedOffset = (samplingZoom_ - 1) * 0.5f * screenSize;
		glm::vec2 offset = baseOffset +
			glm::max(-allowedOffset, glm::min(allowedOffset, foveatedOffset));
		torch::Tensor offsetTensor = torch::tensor({ offset.x, offset.y }, at::dtype(at::kFloat));
		offsetTensor = offsetTensor.unsqueeze(1).to(at::kCUDA);
		samplingScaledPattern_ = samplingPattern_ * factor + offsetTensor;

		//create output
		samplingOutput_ = torch::empty({ numChannels, N },
			torch::dtype(torch::kFloat).device(torch::kCUDA));

		//settings
		renderer::RendererArgs args = setupRendererArgs(renderMode, 1);
		static const int VIEWPORT_BORDER = 100; //overdrawing of samples over the border
		args.cameraViewport = make_int4(-VIEWPORT_BORDER, -VIEWPORT_BORDER,
			displayWidth_ + VIEWPORT_BORDER, displayHeight_ + VIEWPORT_BORDER);
		args.aoSamples = requiresAO ? rendererArgs_.aoSamples : 0;

		//render samples
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		render_samples_gpu(volume_.get(), &args, samplingScaledPattern_, samplingOutput_, stream);
		samplingNumberOfSamplesTaken_ = samplingScaledPattern_.size(1);

		//blend samples into rendererOutput_
		samplingMeshDrawer_->modifyVertexBuffer(samplingScaledPattern_, samplingOutput_);
		//sparse points
		samplingMeshDrawer_->draw(
			{ 2.0f / displayWidth_, 2.0f / displayHeight_ },
			{ -1.0f, -1.0f },
			{ displayWidth_, displayHeight_ },
			true,
			{ 0.5, 0, 0, 0, 0, 0, 0, 0 });
		rendererSparseOutput_ = samplingMeshDrawer_->grabResult();
		rendererSparseOutput_[0] = rendererSparseOutput_[0] * 2 - 1;
		rendererSparseOutput_ = rendererSparseOutput_.unsqueeze(0);
		//interpolated points
		if (samplingInterpolationMode_ == SamplingInterpolationBarycentric)
		{
			samplingMeshDrawer_->draw(
				{ 2.0f / displayWidth_, 2.0f / displayHeight_ },
				{ -1.0f, -1.0f },
				{ displayWidth_, displayHeight_ },
				false,
				{ 0, 0, 0, 0, 0, 0, 0, 0 });
			rendererInterpolatedOutput_ = samplingMeshDrawer_->grabResult();
			rendererInterpolatedOutput_ = rendererInterpolatedOutput_.unsqueeze(0);
		}
		else //inpainting
		{
			const auto mask = at::abs(rendererSparseOutput_.narrow(1, maskChannel, 1));
			rendererInterpolatedOutput_ = kernel::inpaintFlow(mask, rendererSparseOutput_);
		}

		redrawMode_ = RedrawNetwork;
	}

	return RenderAdaptiveSamplesCanReconstruct; //we have data
}

void Visualizer::renderAdaptiveSamples_Reconstruct(
	RenderMode renderMode, AdaptiveReconstructionMethodPtr selectedNetwork)
{
	//configuration between ISO and DVR
	const int numChannels = (renderMode == IsosurfaceRendering)
		? renderer::IsoRendererOutputChannels : renderer::DvrRendererOutputChannels;
	const int maskChannel = (renderMode == IsosurfaceRendering) ? 0 : 3;
	const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;
	
	if (redrawMode_ >= RedrawNetwork)
	{
		if (selectedNetwork->isGroundTruth())
		{
			//special case: ground truth rendering without adaptive samples
			renderer::RendererArgs args = setupRendererArgs(renderMode, 1);
			args.aoSamples = selectedNetwork->requiresAO() ? rendererArgs_.aoSamples : 0;

			cudaStream_t stream = at::cuda::getCurrentCUDAStream();
			networkOutput_ = torch::empty({ numChannels, args.cameraResolutionY, args.cameraResolutionX },
				torch::dtype(torch::kFloat).device(torch::kCUDA));

			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			const auto start = std::chrono::steady_clock::now();
			render_gpu(volume_.get(), &args, networkOutput_, stream);
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			auto end = std::chrono::steady_clock::now();
			while (timeReconstruction_.size() > MAX_TIME_ENTRIES) timeReconstruction_.pop_front();
			timeReconstruction_.push_back(std::chrono::duration<float>(end - start).count());

			networkOutput_ = networkOutput_.unsqueeze(0);
			//rendererInpaintedFlow_ = networkOutput_.narrow(1, flowChannel, 2);
		}
		else
		{
			//the network is used to convert the sparse samples to a dense image

			torch::Tensor previousOutput;
			torch::Tensor rendererInterpolatedOutput = rendererInterpolatedOutput_.clone();
			torch::Tensor rendererSparseOutput = rendererSparseOutput_.clone();
			torch::Tensor rendererSparseMask = rendererSparseMask_.clone();
			if (selectedNetwork->requiresPrevious())
			{
				//the network requires the previous output, so assemble it
				if (previousNetworkOutput_.defined() && 
					previousNetworkOutput_.sizes() == rendererSparseOutput_.sizes() &&
					temporalConsistency_)
				{
					if (samplingForceFlowToZero_)
					{
						//ignore flow from the network
						previousOutput = previousNetworkOutput_;
						auto shape = rendererInterpolatedOutput.sizes().vec();
						shape[1] = 2;
						rendererInpaintedFlow_ = torch::zeros(shape,
							rendererInterpolatedOutput.options());
					}
					else if (selectedNetwork->externalFlow())
					{
						//use interpolated flow from the current frame to warp
						const auto mask = rendererSparseOutput.narrow(1, maskChannel, 1);
						//rendererInapintedFlow_ computed already from low-res
						//rendererInpaintedFlow_ = kernel::inpaintFlow(
						//	mask, rendererSparseOutput.narrow(1, flowChannel, 2));
						rendererInterpolatedOutput = torch::cat({
							rendererInterpolatedOutput_.narrow(1, 0, flowChannel),
							rendererInpaintedFlow_ }, 1);
						previousOutput = kernel::warpUpscale(
							previousNetworkOutput_,
							rendererInpaintedFlow_,
							1);
					}
					else
					{
						//we have to warp the previous output based on the previous flow
						//as input to the current network
						rendererInpaintedFlow_ = previousNetworkOutput_.narrow(1, flowChannel, 2);
						previousOutput = kernel::warpUpscale(
							previousNetworkOutput_,
							rendererInpaintedFlow_,
							1);
					}
				}
				else
				{
					//no previous output available, use zero instead
					previousOutput = torch::zeros_like(rendererSparseOutput);
				}
			}

			//send to network
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			const auto start = std::chrono::steady_clock::now();
			networkOutput_ = selectedNetwork->eval(
				rendererSparseOutput,
				rendererSparseMask,
				rendererInterpolatedOutput,
				previousOutput);
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			auto end = std::chrono::steady_clock::now();
			while (timeReconstruction_.size() > MAX_TIME_ENTRIES) timeReconstruction_.pop_front();
			timeReconstruction_.push_back(std::chrono::duration<float>(end - start).count());
		}

		//save output for the next round
		previousNetworkOutput_ = networkOutput_;

		redrawMode_ = RedrawPost;
	}

	if (redrawMode_ == RedrawPost)
	{
		//temporal reprojection
		torch::Tensor blendingOutput;
		if (!previousBlendingOutput_.defined() ||
			previousBlendingOutput_.sizes() != networkOutput_.sizes() ||
			temporalPostSmoothingPercentage_ == 0)
		{
			blendingOutput = networkOutput_;
		}
		else
		{
			auto previousOutput = kernel::warpUpscale(
				previousBlendingOutput_,
				rendererInpaintedFlow_,
				1);
			float blendingFactor = temporalPostSmoothingPercentage_ / 100.0f;
			blendingOutput = blendingFactor * previousOutput
				+ (1 - blendingFactor)*networkOutput_;
		}
		//channel selection
		if (renderMode == IsosurfaceRendering)
			selectChannelIso(channelMode_, blendingOutput, screenTextureCudaBuffer_);
		else
			selectChannelDvr(channelMode_, blendingOutput, screenTextureCudaBuffer_);
		previousBlendingOutput_ = blendingOutput;
		copyBufferToOpenGL();

		redrawMode_ = RedrawNone;
	}
}

void Visualizer::renderAdaptiveStepsize(RenderMode renderMode)
{
	if (renderMode != DirectVolumeRendering)
		return;

	auto selectedReconstructionNetwork = stepsizeReconstructionDvr_[selectedStepsizeReconstructionDvr_];
	auto selectedImportanceNetwork = stepsizeImportanceSamplerDvr_[selectedStepsizeImportanceSamplerDvr_];
	
	int result = renderAdaptiveSamples_SampleDynamic(renderMode, selectedImportanceNetwork, false);
	if (result == RenderAdaptiveSamplesNeedsSampling)
		result = renderAdaptiveStepsize_RenderSamples(renderMode, selectedReconstructionNetwork->requiresAO());

	if (result == RenderAdaptiveSamplesCanReconstruct ||
		((result == RenderAdaptiveSamplesNoData) && selectedReconstructionNetwork->isGroundTruth()))
	{
		renderAdaptiveStepsize_Reconstruct(renderMode, selectedReconstructionNetwork);
		drawer_.drawQuad(screenTextureGL_);
	}
	else if (result == RenderAdaptiveSamplesAlreadyFilled)
	{
		drawer_.drawQuad(screenTextureGL_);
	}

	redrawMode_ = RedrawNone;
}

int Visualizer::renderAdaptiveStepsize_RenderSamples(RenderMode renderMode, bool requiresAO)
{
	//6. render samples
	//configuration between ISO and DVR
	const int numChannels = (renderMode == IsosurfaceRendering)
		? renderer::IsoRendererOutputChannels : renderer::DvrRendererOutputChannels;
	const int maskChannel = (renderMode == IsosurfaceRendering) ? 0 : 3;
	const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;

	//create output
	rendererOutput_ = torch::empty(
		{ numChannels, displayHeight_, displayWidth_ },
		torch::dtype(torch::kFloat).device(torch::kCUDA));

	//settings
	renderer::RendererArgs args = setupRendererArgs(renderMode);
	args.cameraResolutionX = displayWidth_;
	args.cameraResolutionY = displayHeight_;
	args.cameraViewport = make_int4(0, 0, displayWidth_, displayHeight_);
	args.aoSamples = requiresAO ? rendererArgs_.aoSamples : 0;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	torch::Tensor stepsize = 1.0f / normalizedImportanceMap_;
	renderer::render_adaptive_stepsize_gpu(volume_.get(), &args, stepsize, rendererOutput_, stream);
	rendererOutput_ = rendererOutput_.unsqueeze(0);
	samplingNumberOfSamplesTaken_ = 1;
	averageSamplesPerVoxel_ = normalizedImportanceMap_.mean().item<float>() / args.stepsize;
	minAdaptiveStepsize_ = stepsize.min().item<float>() * args.stepsize;
	maxAdaptiveStepsize_ = stepsize.max().item<float>() * args.stepsize;

	//done
	redrawMode_ = RedrawNetwork;
	return RenderAdaptiveSamplesCanReconstruct;
}

void Visualizer::renderAdaptiveStepsize_Reconstruct(RenderMode renderMode, AdaptiveStepsizeReconstructionMethodPtr selectedNetwork)
{
	//configuration between ISO and DVR
	const int numChannels = (renderMode == IsosurfaceRendering)
		? renderer::IsoRendererOutputChannels : renderer::DvrRendererOutputChannels;
	const int maskChannel = (renderMode == IsosurfaceRendering) ? 0 : 3;
	const int flowChannel = (renderMode == IsosurfaceRendering) ? 6 : 8;

	if (redrawMode_ >= RedrawNetwork)
	{
		if (selectedNetwork->isGroundTruth())
		{
			//special case: ground truth rendering without adaptive samples
			renderer::RendererArgs args = setupRendererArgs(renderMode, 1);
			args.aoSamples = selectedNetwork->requiresAO() ? rendererArgs_.aoSamples : 0;

			cudaStream_t stream = at::cuda::getCurrentCUDAStream();
			networkOutput_ = torch::empty({ numChannels, args.cameraResolutionY, args.cameraResolutionX },
				torch::dtype(torch::kFloat).device(torch::kCUDA));
			render_gpu(volume_.get(), &args, networkOutput_, stream);
			networkOutput_ = networkOutput_.unsqueeze(0);
			rendererInpaintedFlow_ = networkOutput_.narrow(1, flowChannel, 2);
			averageSamplesPerVoxel_ = 1.0f / args.stepsize;
		}
		else
		{
			//the network is used to convert the sparse samples to a dense image

			torch::Tensor previousOutput;
			torch::Tensor rendererOutput = rendererOutput_.clone();
			std::cout << "Renderer output size:" << rendererOutput.sizes() << std::endl;
			if (selectedNetwork->requiresPrevious())
			{
				//the network requires the previous output, so assemble it
				if (previousNetworkOutput_.defined() &&
					previousNetworkOutput_.sizes() == rendererOutput.sizes() &&
					temporalConsistency_)
				{
					if (samplingForceFlowToZero_)
					{
						//ignore flow from the network
						previousOutput = previousNetworkOutput_;
						auto shape = rendererOutput.sizes().vec();
						shape[1] = 2;
						rendererInpaintedFlow_ = torch::zeros(shape,
							rendererOutput.options());
					}
					else
					{
						//use interpolated flow from the current frame to warp
						const auto mask = rendererOutput.narrow(1, maskChannel, 1);
						rendererInpaintedFlow_ = kernel::inpaintFlow(
							mask, rendererOutput.narrow(1, flowChannel, 2));
						rendererOutput = torch::cat({
							rendererOutput.narrow(1, 0, flowChannel),
							rendererInpaintedFlow_ }, 1);
						previousOutput = kernel::warpUpscale(
							previousNetworkOutput_,
							rendererInpaintedFlow_,
							1);
					}
				}
				else
				{
					//no previous output available, use zero instead
					previousOutput = torch::zeros_like(rendererOutput);
				}
			}

			//send to network
			networkOutput_ = selectedNetwork->eval(
				rendererOutput,
				previousOutput);
		}

		//save output for the next round
		previousNetworkOutput_ = networkOutput_;

		redrawMode_ = RedrawPost;
	}

	if (redrawMode_ == RedrawPost)
	{
		//temporal reprojection
		torch::Tensor blendingOutput;
		if (!previousBlendingOutput_.defined() ||
			previousBlendingOutput_.sizes() != networkOutput_.sizes() ||
			temporalPostSmoothingPercentage_ == 0)
		{
			blendingOutput = networkOutput_;
		}
		else
		{
			auto previousOutput = kernel::warpUpscale(
				previousBlendingOutput_,
				rendererInpaintedFlow_,
				1);
			float blendingFactor = temporalPostSmoothingPercentage_ / 100.0f;
			blendingOutput = blendingFactor * previousOutput
				+ (1 - blendingFactor)*networkOutput_;
		}
		//channel selection
		if (renderMode == IsosurfaceRendering)
			selectChannelIso(channelMode_, blendingOutput, screenTextureCudaBuffer_);
		else
			selectChannelDvr(channelMode_, blendingOutput, screenTextureCudaBuffer_);
		previousBlendingOutput_ = blendingOutput;
		copyBufferToOpenGL();

		redrawMode_ = RedrawNone;
	}
}

void Visualizer::renderSuperresolutionIso()
{
	//select the network to use, mind foveated rendering
	SuperresolutionMethodPtr selectedNetwork = networksIso_[selectedNetworkIso_];
	if (foveatedEnable_)
	{
		for (auto it = foveatedLayers_.rbegin(); it != foveatedLayers_.rend(); ++it)
		{
			if (it->method_ != nullptr) {
				selectedNetwork = it->method_;
				break;
			}
		}
	}

	//render iso-surface to rendererOutput_
	if (redrawMode_ == RedrawRenderer)
	{
		int upscale_factor = selectedNetwork->supportsArbitraryUpscaleFactors()
			? superresUpscaleFactor_
			: selectedNetwork->upscaleFactor();
		renderer::RendererArgs args = setupRendererArgs(IsosurfaceRendering, upscale_factor);
		args.aoSamples = selectedNetwork->requiresAO() ? rendererArgs_.aoSamples : 0;

		rendererOutput_ = torch::empty({ renderer::IsoRendererOutputChannels, args.cameraResolutionY, args.cameraResolutionX },
			torch::dtype(torch::kFloat).device(torch::kCUDA));

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		render_gpu(volume_.get(), &args, rendererOutput_, stream);
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		rendererOutput_ = rendererOutput_.unsqueeze(0);
		rendererInpaintedFlow_ = torch::Tensor();

		redrawMode_ = RedrawNetwork;
	}

	//flow inpainting
	bool needFlow =
		(selectedNetwork->requiresPrevious() && temporalConsistency_) ||
		temporalPostSmoothingPercentage_ > 0 ||
		(channelMode_ == ChannelFlow && flowWithInpainting_);
	if (needFlow && !rendererInpaintedFlow_.defined())
	{
		auto flow = rendererOutput_.narrow(1, 6, 2);
		auto mask = rendererOutput_.narrow(1, 0, 1);
		rendererInpaintedFlow_ = kernel::inpaintFlow(mask, flow);
	}

	//super-resolution
	if (redrawMode_ == RedrawNetwork)
	{
		torch::Tensor previousOutput = {};
		if (selectedNetwork->requiresPrevious() && temporalConsistency_)
		{
			previousOutput = kernel::warpUpscale(
				previousNetworkOutput_,
				rendererInpaintedFlow_,
				selectedNetwork->upscaleFactor());
		}

		int upscaleFactor = selectedNetwork->supportsArbitraryUpscaleFactors()
			? superresUpscaleFactor_
			: 0;
		networkOutput_ = selectedNetwork->eval(rendererOutput_, previousOutput, upscaleFactor);
		redrawMode_ = RedrawPost;

		previousNetworkOutput_ = networkOutput_.detach();

		//pad to get the correct screen size
		//needed if displayWidth_/displayHeight_ are not multiples of the upscaling factor
		int padW = displayWidth_ - networkOutput_.size(3);
		int padH = displayHeight_ - networkOutput_.size(2);
		if (padW != 0 || padH != 0)
			networkOutput_ = torch::constant_pad_nd(networkOutput_,
				{ 0, padW, 0, padH }, 0);
	}

	//select channel and write to screen texture
	//this also includes the temporal reprojection
	if (redrawMode_ == RedrawPost)
	{
		if (channelMode_ == ChannelFlow)
		{
			if (flowWithInpainting_)
			{
				auto flow = kernel::interpolateNearest(rendererInpaintedFlow_, displayWidth_, displayHeight_);
				kernel::selectOutputChannel(flow, postOutput_,
					0, 1, -1, -2,
					10, 0.5, 1, 0);
			}
			else
			{
				auto flow = kernel::interpolateLinear(rendererOutput_, displayWidth_, displayHeight_);
				kernel::selectOutputChannel(flow, postOutput_,
					6, 7, -1, -2,
					10, 0.5, 1, 0);
			}
		}
		else {
			//temporal reprojection
			torch::Tensor blendingOutput;
			if (!previousBlendingOutput_.defined() ||
				previousBlendingOutput_.sizes() != networkOutput_.sizes() ||
				temporalPostSmoothingPercentage_ == 0)
			{
				blendingOutput = networkOutput_;
			}
			else
			{
				auto previousOutput = kernel::warpUpscale(
					previousBlendingOutput_,
					rendererInpaintedFlow_,
					selectedNetwork->upscaleFactor());
				float blendingFactor = temporalPostSmoothingPercentage_ / 100.0f;
				blendingOutput = blendingFactor * previousOutput
					+ (1 - blendingFactor)*networkOutput_;
			}
			//channel selection
			selectChannelIso(channelMode_, blendingOutput, postOutput_);
			previousBlendingOutput_ = blendingOutput;
		}

		redrawMode_ = RedrawFoveated;
	}

	cudaMemcpy(screenTextureCudaBuffer_, postOutput_, 4 * displayWidth_*displayHeight_,
		cudaMemcpyDeviceToDevice);
	if (redrawMode_ == RedrawFoveated && foveatedEnable_)
	{
		renderFoveated();
		redrawMode_ = RedrawNone;
	}
	else if (redrawMode_ == RedrawFoveated)
		redrawMode_ = RedrawNone;

	//copy screenTextureCudaBuffer_ to screenTextureCuda_
	copyBufferToOpenGL();

	//draw texture
	drawer_.drawQuad(screenTextureGL_);
}

void Visualizer::renderSuperresolutionDvr()
{
	//select the network to use, mind foveated rendering
	SuperresolutionMethodPtr selectedNetwork = networksDvr_[selectedNetworkDvr_];
	if (foveatedEnable_)
	{
		for (auto it = foveatedLayers_.rbegin(); it != foveatedLayers_.rend(); ++it)
		{
			if (it->method_ != nullptr)
			{
				selectedNetwork = it->method_;
				break;
			}
		}
	}

	//render iso-surface to rendererOutput_
	if (redrawMode_ == RedrawRenderer)
	{
		int upscaleFactor = selectedNetwork->supportsArbitraryUpscaleFactors()
			? superresUpscaleFactor_
			: selectedNetwork->upscaleFactor();
		renderer::RendererArgs args = setupRendererArgs(DirectVolumeRendering, upscaleFactor);

		rendererOutput_ = torch::empty({ renderer::DvrRendererOutputChannels, args.cameraResolutionY, args.cameraResolutionX },
			torch::dtype(torch::kFloat).device(torch::kCUDA));

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		render_gpu(volume_.get(), &args, rendererOutput_, stream);
		rendererOutput_ = rendererOutput_.unsqueeze(0);

		redrawMode_ = RedrawNetwork;
	}

	//super-resolution
	if (redrawMode_ == RedrawNetwork)
	{
		torch::Tensor previousOutput = {};
		if (selectedNetwork->requiresPrevious() && temporalConsistency_ && 
			previousNetworkOutput_.defined())
		{
			previousOutput = kernel::warpUpscale(
				previousNetworkOutput_,
				rendererInpaintedFlow_,
				selectedNetwork->upscaleFactor());
		}
		
		int upscaleFactor = selectedNetwork->supportsArbitraryUpscaleFactors()
			? superresUpscaleFactor_
			: 0;
		networkOutput_ = selectedNetwork->eval(rendererOutput_, previousOutput, upscaleFactor);
		redrawMode_ = RedrawPost;

		previousNetworkOutput_ = networkOutput_.detach();
		
		//pad to get the correct screen size
		//needed if displayWidth_/displayHeight_ are not multiples of the upscaling factor
		int padW = displayWidth_ - networkOutput_.size(3);
		int padH = displayHeight_ - networkOutput_.size(2);
		if (padW != 0 || padH != 0)
			networkOutput_ = torch::constant_pad_nd(networkOutput_,
				{ 0, padW, 0, padH }, 0);
	}

	//select channel and write to screen texture
	//this also includes the temporal reprojection
	if (redrawMode_ == RedrawPost)
	{
		if (channelMode_ == ChannelFlow)
		{
			auto flow = kernel::interpolateLinear(rendererOutput_, displayWidth_, displayHeight_);
			kernel::selectOutputChannel(flow, postOutput_,
				8, 9, -1, -2,
				10, 0.5, 1, 0);
		}
		else {
			//temporal reprojection
			torch::Tensor blendingOutput;
			if (!previousBlendingOutput_.defined() ||
				previousBlendingOutput_.sizes() != networkOutput_.sizes() ||
				temporalPostSmoothingPercentage_ == 0)
			{
				blendingOutput = networkOutput_;
			}
			else
			{
				auto previousOutput = kernel::warpUpscale(
					previousBlendingOutput_,
					rendererOutput_.narrow(1, 8, 2),
					selectedNetwork->upscaleFactor());
				float blendingFactor = temporalPostSmoothingPercentage_ / 100.0f;
				blendingOutput = blendingFactor * previousOutput
					+ (1 - blendingFactor)*networkOutput_;
			}
			//channel selection
			selectChannelDvr(channelMode_, blendingOutput, postOutput_);
			previousBlendingOutput_ = blendingOutput;
		}

		redrawMode_ = RedrawNone;
	}
	
	//kernel::transferDvrOutput(networkOutput_, screenTextureCudaBuffer_, displayWidth_, displayHeight_);
	cudaMemcpy(screenTextureCudaBuffer_, postOutput_, 4 * displayWidth_*displayHeight_,
		cudaMemcpyDeviceToDevice);
	
	//copy screenTextureCudaBuffer_ to screenTextureCuda_
	copyBufferToOpenGL();

	//draw texture
	drawer_.drawQuad(screenTextureGL_);
}

void Visualizer::renderFoveated()
{
	//we have to render + superresolution + blend all foveated layers

	//first test, write everything into temporal buffers
	//TODO: refactor duplicate code with drawFoveatedMask
	std::vector<kernel::LayerData> layers;
	float radiusToPixel = glm::length(glm::vec2(displayWidth_, displayHeight_));
	for (int i = 0; i < foveatedLayers_.size() - 1; ++i)
	{
		//prepare layer
		kernel::LayerData layer;
		const auto& d = foveatedLayers_[foveatedLayers_.size() - i - 2];
		if (d.method_ == nullptr) continue; //disabled
		int upscaleFactor = d.method_->upscaleFactor();
		//std::cout << "Layer " << i << ", factor=" << upscaleFactor << std::endl;
		layer.subimage = nullptr;
		layer.blurShape = foveatedBlurShape_;
		layer.viewport.x = static_cast<int>(foveatedCenter_.x - d.radius_*0.01f*radiusToPixel*0.5f) / upscaleFactor * upscaleFactor;
		layer.viewport.y = static_cast<int>(foveatedCenter_.y - d.radius_*0.01f*radiusToPixel*0.5f) / upscaleFactor * upscaleFactor;
		layer.viewport.z = static_cast<int>(d.radius_*0.01f*radiusToPixel) / upscaleFactor * upscaleFactor;
		layer.viewport.w = static_cast<int>(d.radius_*0.01f*radiusToPixel) / upscaleFactor * upscaleFactor;
		int prevRadius = (i < (foveatedLayers_.size() - 2)) ? foveatedLayers_[foveatedLayers_.size() - i - 3].radius_ : 0;
		layer.smoothingRadius = (d.radius_ - prevRadius)*fovatedBlurRadiusPercent_*0.01*0.01*radiusToPixel / layer.viewport.z;
		//std::cout << "high-viewport = (" <<
		//	layer.viewport.x << ", " << layer.viewport.y << ", " <<
		//	layer.viewport.z << ", " << layer.viewport.w << ")" << std::endl;

		//render
		renderer::RendererArgs args = rendererArgs_;
		args.cameraResolutionX = displayWidth_ / upscaleFactor;
		args.cameraResolutionY = displayHeight_ / upscaleFactor;
		args.aoSamples = d.method_->requiresAO() ? rendererArgs_.aoSamples : 0;
		args.cameraViewport = make_int4(
			layer.viewport.x / upscaleFactor,
			layer.viewport.y / upscaleFactor,
			layer.viewport.z / upscaleFactor + layer.viewport.x / upscaleFactor,
			layer.viewport.w / upscaleFactor + layer.viewport.y / upscaleFactor);
		args.mipmapLevel = volumeMipmapLevel_;
		//std::cout << "low-viewport = (" <<
		//	args.cameraViewport.x << ", " << args.cameraViewport.y << ", " <<
		//	args.cameraViewport.z << ", " << args.cameraViewport.w << ")" << std::endl;
		torch::Tensor rendererOutput = torch::empty(
			{
				renderer::IsoRendererOutputChannels,
				args.cameraViewport.w - args.cameraViewport.y,
				args.cameraViewport.z - args.cameraViewport.x },
				torch::dtype(torch::kFloat).device(torch::kCUDA));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		render_gpu(volume_.get(), &args, rendererOutput, stream);
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		rendererOutput = rendererOutput.unsqueeze(0);
		//std::cout << "render output = " << rendererOutput.sizes() << std::endl;

		//superresolution
		//TODO: temporal consistency
		torch::Tensor previousOutput = {};
		int localUpscaleFactor = d.method_->supportsArbitraryUpscaleFactors()
			? d.method_->upscaleFactor()
			: 0;
		torch::Tensor networkOutput = d.method_->eval(rendererOutput, previousOutput, localUpscaleFactor);
		//std::cout << "network output = " << networkOutput.sizes() << std::endl;

		//alloc output buffer and select channel
		CUMAT_SAFE_CALL(cudaMalloc(&layer.subimage, 4 * layer.viewport.z * layer.viewport.w));
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		selectChannelIso(channelMode_, networkOutput, layer.subimage);

		layers.push_back(layer);
	}
	kernel::foveatedBlending(displayWidth_, displayHeight_, screenTextureCudaBuffer_, layers);
	//cleanup temporar buffers
	for (auto& d : layers)
		CUMAT_SAFE_CALL(cudaFree(d.subimage));
}

void Visualizer::copyBufferToOpenGL()
{
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &screenTextureCuda_, 0));
	cudaArray* texture_ptr;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, screenTextureCuda_, 0, 0));
	size_t size_tex_data = sizeof(GLubyte) * displayWidth_ * displayHeight_ * 4;
	CUMAT_SAFE_CALL(cudaMemcpyToArray(texture_ptr, 0, 0, screenTextureCudaBuffer_, size_tex_data, cudaMemcpyDeviceToDevice));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &screenTextureCuda_, 0));
}

void Visualizer::resize(int display_w, int display_h)
{
	//make it a nice multiplication of everything
	const int multiply = 4 * 3;
	display_w = display_w / multiply * multiply;
	display_h = display_h / multiply * multiply;

	if (display_w == displayWidth_ && display_h == displayHeight_)
		return;
	if (display_w == 0 || display_h == 0)
		return;
	releaseResources();
	displayWidth_ = display_w;
	displayHeight_ = display_h;

	//create texture
	glGenTextures(1, &screenTextureGL_);
	glBindTexture(GL_TEXTURE_2D, screenTextureGL_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA8,
		displayWidth_, displayHeight_, 0
		, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	//register with cuda
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(
		&screenTextureCuda_, screenTextureGL_,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//create channel output buffer
	CUMAT_SAFE_CALL(cudaMalloc(&screenTextureCudaBuffer_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));
	CUMAT_SAFE_CALL(cudaMalloc(&postOutput_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));

	glBindTexture(GL_TEXTURE_2D, 0);

	triggerRedraw(RedrawRenderer);
	std::cout << "Visualizer::resize(): " << displayWidth_ << ", " << displayHeight_ << std::endl;
}

void Visualizer::triggerRedraw(RedrawMode mode)
{
	redrawMode_ = std::max(redrawMode_, mode);
}

void Visualizer::selectChannelIso(ChannelMode mode, const torch::Tensor& networkOutput, GLubyte* cudaBuffer) const
{
	if (networkOutput.size(2) != displayHeight_ || networkOutput.size(3) != displayWidth_)
	{
		std::cout << "Tensor shape does not match: expected=(1 * Channels * "
			<< displayHeight_ << " * " << displayWidth_ << "), but got: "
			<< networkOutput.sizes() << std::endl;
		throw std::exception("Tensor shape does not match screen size");
	}
	switch (mode)
	{
	case ChannelMask:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			0, 0, 0, 0,
			1, 0, 0, 1);
		break;
	case ChannelNormal:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			1, 2, 3, 0,
			0.5, 0.5, 1, 0);
		break;
	case ChannelDepth: {
		auto depthVal = networkOutput[0][4];
		float maxDepth = *at::max(depthVal).to(at::kCPU).data<float>();
		float minDepth = *at::min(depthVal + at::le(depthVal, 1e-5).type_as(depthVal)).to(at::kCPU).data<float>();
		//std::cout << "depth: min=" << minDepth << ", max=" << maxDepth << std::endl;
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			4, 4, 4, 0,
			1 / (maxDepth - minDepth), -minDepth / (maxDepth - minDepth), 1, 0);
		break;
	}
	case ChannelAO:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			5, 5, 5, 0,
			1, 0, 1, 0);
		break;
	case ChannelFlow:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			6, 7, -1, -2,
			10, 0.5, 1, 0);
		break;
	case ChannelColor: {
		renderer::ShadingSettings settings;
		settings.ambientLightColor = ambientLightColor;
		settings.diffuseLightColor = diffuseLightColor;
		settings.specularLightColor = specularLightColor;
		settings.specularExponent = specularExponent;
		settings.materialColor = materialColor;
		settings.aoStrength = aoStrength;
		settings.lightDirection = lightDirectionScreen;
		kernel::screenShading(networkOutput, cudaBuffer, settings);
		break;
	}
	default:
		throw std::exception("unknown enum");
	}
}

void Visualizer::selectChannelDvr(ChannelMode mode, const torch::Tensor& networkOutput, GLubyte* cudaBuffer) const
{
	if (networkOutput.size(2) != displayHeight_ || networkOutput.size(3) != displayWidth_)
	{
		std::cout << "Tensor shape does not match: expected=(1 * Channels * "
			<< displayHeight_ << " * " << displayWidth_ << "), but got: "
			<< networkOutput.sizes() << std::endl;
		throw std::exception("Tensor shape does not match screen size");
	}
	switch (mode)
	{
	case ChannelColor:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			0, 1, 2, 3,
			1, 0, 1, 0);
		break;
	case ChannelMask:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			3, 3, 3, 3,
			1, 0, 0, 1);
		break;
	case ChannelNormal:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			4, 5, 6, 3,
			0.5, 0.5, 1, 0);
		break;
	case ChannelDepth: {
		auto depthVal = networkOutput[0][7];
		float maxDepth = *at::max(depthVal).to(at::kCPU).data<float>();
		float minDepth = *at::min(depthVal + at::le(depthVal, 1e-5).type_as(depthVal)).to(at::kCPU).data<float>();
		//std::cout << "depth: min=" << minDepth << ", max=" << maxDepth << std::endl;
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			7, 7, 7, 3,
			1 / (maxDepth - minDepth), -minDepth / (maxDepth - minDepth), 1, 0);
		break;
	}
	case ChannelAO:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			-1, -1, -1, -1, //disabled
			1, 0, 1, 0);
		break;
	case ChannelFlow:
		kernel::selectOutputChannel(networkOutput, cudaBuffer,
			8, 9, -1, -2,
			10, 0.5, 1, 0);
		break;
	default:
		throw std::exception("unknown enum");
	}
}

void Visualizer::drawFoveatedMask(GLubyte* cudaBuffer) const
{
	cudaMemset(cudaBuffer, 0, 4 * displayWidth_*displayHeight_);
	std::vector<kernel::LayerData> layers;
	float radiusToPixel = glm::length(glm::vec2(displayWidth_, displayHeight_));
	for (int i = 0; i < foveatedLayers_.size() - 1; ++i)
	{
		kernel::LayerData layer;
		const auto& d = foveatedLayers_[foveatedLayers_.size() - i - 2];
		if (d.method_ == nullptr) continue; //disabled
		layer.subimage = nullptr;
		layer.blurShape = foveatedBlurShape_;
		layer.viewport.x = static_cast<int>(foveatedCenter_.x - d.radius_*0.01f*radiusToPixel*0.5f);
		layer.viewport.y = static_cast<int>(foveatedCenter_.y - d.radius_*0.01f*radiusToPixel*0.5f);
		layer.viewport.z = static_cast<int>(d.radius_*0.01f*radiusToPixel);
		layer.viewport.w = static_cast<int>(d.radius_*0.01f*radiusToPixel);
		int prevRadius = (i < (foveatedLayers_.size() - 2)) ? foveatedLayers_[foveatedLayers_.size() - i - 3].radius_ : 0;
		layer.smoothingRadius = (d.radius_ - prevRadius)*fovatedBlurRadiusPercent_*0.01*0.01*radiusToPixel / layer.viewport.z;
		//std::cout << "Layer " << i << ": viewport=(" <<
		//	layer.viewport.x << ", " << layer.viewport.y << ", " <<
		//	layer.viewport.z << ", " << layer.viewport.w << "), smoothing=" <<
		//	layer.smoothingRadius << std::endl;
		layers.push_back(layer);
	}
	kernel::foveatedBlending(displayWidth_, displayHeight_, cudaBuffer, layers);
}

float Visualizer::foveatedComputeNumberOfSamples()
{
	//ground truth samples
	int fullResSamples = displayWidth_ * displayHeight_;

	//outer, screen filling layer
	SuperresolutionMethodPtr outerNetwork = nullptr;
	for (auto it = foveatedLayers_.rbegin(); it != foveatedLayers_.rend(); ++it)
	{
		if (it->method_ != nullptr) {
			outerNetwork = it->method_;
			break;
		}
	}
	int lowResSamples = fullResSamples / outerNetwork->upscaleFactor() / outerNetwork->upscaleFactor();

	//inner layers
	int innerSamples = 0;
	float radiusToPixel = glm::length(glm::vec2(displayWidth_, displayHeight_));
	for (int i = 0; i < foveatedLayers_.size() - 1; ++i)
	{
		const auto& d = foveatedLayers_[foveatedLayers_.size() - i - 2];
		if (d.method_ == nullptr) continue; //disabled
		int upscaleFactor = d.method_->upscaleFactor();
		//compute viewport
		int4 viewport;
		viewport.x = static_cast<int>(foveatedCenter_.x - d.radius_*0.01f*radiusToPixel*0.5f) / upscaleFactor * upscaleFactor;
		viewport.y = static_cast<int>(foveatedCenter_.y - d.radius_*0.01f*radiusToPixel*0.5f) / upscaleFactor * upscaleFactor;
		viewport.z = viewport.x + static_cast<int>(d.radius_*0.01f*radiusToPixel) / upscaleFactor * upscaleFactor;
		viewport.w = viewport.y + static_cast<int>(d.radius_*0.01f*radiusToPixel) / upscaleFactor * upscaleFactor;
		//clamp at the screen boundaries
		viewport.x = std::max(0, viewport.x);
		viewport.y = std::max(0, viewport.y);
		viewport.z = std::min(displayWidth_ - 1, viewport.z);
		viewport.w = std::min(displayHeight_ - 1, viewport.w);
		//compute samples and accumulate
		int samples = (viewport.z - viewport.x) * (viewport.w - viewport.y) /
			upscaleFactor / upscaleFactor;
		innerSamples += samples;
	}

	return (innerSamples + lowResSamples) / float(fullResSamples);
}

void Visualizer::screenshot()
{
	std::string folder = "screenshots";

	char time_str[128];
	time_t now = time(0);
	struct tm tstruct;
	localtime_s(&tstruct, &now);
	strftime(time_str, sizeof(time_str), "%Y%m%d-%H%M%S", &tstruct);

	char output_name[512];
	sprintf(output_name, "%s/screenshot_%s_%s.png", folder.c_str(), time_str, ChannelModeNames[channelMode_]);

	std::cout << "Take screenshot: " << output_name << std::endl;
	std::experimental::filesystem::create_directory(folder);

	std::vector<GLubyte> textureCpu(4 * displayWidth_ * displayHeight_);
	CUMAT_SAFE_CALL(cudaMemcpy(&textureCpu[0], screenTextureCudaBuffer_, 4 * displayWidth_*displayHeight_, cudaMemcpyDeviceToHost));

	if (lodepng_encode32_file(output_name, textureCpu.data(), displayWidth_, displayHeight_) != 0)
	{
		std::cerr << "Unable to save image" << std::endl;
		screenshotString_ = std::string("Unable to save screenshot to ") + output_name;
	}
	else
	{
		screenshotString_ = std::string("Screenshot saved to ") + output_name;
	}
	screenshotTimer_ = 2.0f;
}
