#pragma once

#include <cuda_runtime.h>
#include <lib.h>
#include <GL/glew.h>
#include <sstream>
#include <queue>

#include "quad_drawer.h"
#include "tf_editor.h"
#include "camera_gui.h"
#include "superres_model.h"
#include "visualizer_kernels.h"
#include "background_worker.h"
#include "adaptive_model.h"
#include "importance_sampling_model.h"
#include "stepsize_model.h"
#include "visualizer_commons.h"

struct GLFWwindow;

class Visualizer
{
public:
	Visualizer(GLFWwindow* window);
	~Visualizer();

	void specifyUI();

	void render(int display_w, int display_h);

private:

	enum RedrawMode
	{
		RedrawNone,
		RedrawFoveated,
		RedrawPost,
		RedrawNetwork,
		RedrawRenderer,

		_RedrawModeCount_
	};
	static const char* RedrawModeNames[_RedrawModeCount_];
	RedrawMode redrawMode_ = RedrawNone;

	GLFWwindow* window_;

	//volume
	std::string volumeDirectory_;
	std::string volumeFilename_;
	std::string volumeFullFilename_;
	std::unique_ptr<renderer::Volume> volume_;
	static constexpr int MipmapLevels[] = { 0, 1, 2, 3, 7 };
	int volumeMipmapLevel_ = 0;
	renderer::Volume::MipmapFilterMode volumeMipmapFilterMode_
		= renderer::Volume::MipmapFilterMode::AVERAGE;
	renderer::RendererArgs rendererArgs_;

	CameraGui cameraGui_;

	//background computation
	BackgroundWorker worker_;
	std::function<void()> backgroundGui_;

	//information string that is displayed together with the FPS
	//It is cleared every frame. Use it to append per-frame information
	std::stringstream extraFrameInformation_;

	ComputationMode computationMode_ = ComputeAdaptiveStepsize;
	RenderMode renderMode_ = DirectVolumeRendering;

	//display
	int displayWidth_ = 0;
	int displayHeight_ = 0;
	unsigned int screenTextureGL_ = 0;
	cudaGraphicsResource_t screenTextureCuda_ = nullptr;
	GLubyte* screenTextureCudaBuffer_ = nullptr;
	QuadDrawer drawer_;

	//dvr
	TfEditor editor_;
	std::string tfDirectory_;
	float minDensity_{ 0.0f };
	float maxDensity_{ 1.0f };
	float opacityScaling_{ 50.0f };
	bool showColorControlPoints_{ true };
	bool dvrUseShading_ = false;
	RENDERER_NAMESPACE Volume::Histogram volumeHistogram_;

	//intermediate computation results
	torch::Tensor rendererOutput_;
	torch::Tensor rendererSparseOutput_;
	torch::Tensor rendererSparseMask_;
	torch::Tensor rendererInterpolatedOutput_;
	torch::Tensor rendererInpaintedFlow_;
	torch::Tensor networkOutput_;
	torch::Tensor previousNetworkOutput_;
	torch::Tensor previousImportanceNetworkOutput_;
	torch::Tensor previousBlendingOutput_;
	torch::Tensor normalizedImportanceMap_;
	GLubyte* postOutput_ = nullptr;

	//network specs
	std::string networkDirectoryIso_;
	std::vector<SuperresolutionMethodPtr> networksIso_;
	int selectedNetworkIso_ = 0;
	std::string networkDirectoryDvr_;
	std::vector<SuperresolutionMethodPtr> networksDvr_;
	int selectedNetworkDvr_ = 0;
	bool temporalConsistency_ = true;
	int superresUpscaleFactor_ = 4;

	//ADAPTIVE SAMPLING
	//tensors with intermediate results
	torch::Tensor samplingOutput_;
	torch::Tensor importanceMap_;
	int samplingNumberOfSamplesTaken_ = 0; //number of samples taken
	//sampling strategy selection
	enum SamplingStrategy
	{
		///calls a network that generates an importance map from a low-res rendering
		SamplingStrategyDynamic, 
		///fixed sampling mask, e.g. for foveated rendering
		///supports scaling
		SamplingStrategyFixed
	};
	SamplingStrategy samplingStrategy_ = SamplingStrategyDynamic;
	//1. importance map generation
	//dynamic sampling mask
	std::string importanceNetDirectoryIso_;
	std::string importanceNetDirectoryDvr_;
	std::vector<ImportanceSamplingMethodPtr> importanceSamplerIso_;
	std::vector<ImportanceSamplingMethodPtr> importanceSamplerDvr_;
	int selectedImportanceSamplerIso_ = 0;
	int selectedImportanceSamplerDvr_ = 0;
	int importanceUpscale_ = 4;
	bool importanceTestShowInput_ = false;
	bool importanceTestShowImportance_ = false;
	//2. sample generation
	//dynamic sampling mask
	std::string samplingSequenceDirectory_;
	std::string samplingSequenceFilename_;
	std::string samplingSequenceFullFilename_;
	torch::Tensor samplingSequence_; //tensor that maps importance to sample or not
	float samplingMinImportance_ = 0.05;
	float samplingMeanImportance_ = 0.2;
	//fixed sampling mask
	std::string samplingPatternDirectory_;
	std::string samplingPatternFilename_;
	torch::Tensor samplingPattern_;
	torch::Tensor samplingScaledPattern_;
	int2 samplingPatternSize_ = make_int2(0, 0);
	enum SamplingInterpolationMode
	{
		SamplingInterpolationBarycentric, //barycentric, only for precomputed masks
		SamplingInterpolaionInpainting, //simple, fast inpainting, for adaptive masks
		//TODO: PDE-Inpainting
	};
	SamplingInterpolationMode samplingInterpolationMode_ = SamplingInterpolaionInpainting;
	float samplingZoom_ = 1.0;
	std::unique_ptr<renderer::MeshDrawer> samplingMeshDrawer_;
	//3. sampling reconstruction networks
	std::string sparseNetworkDirectoryIso_;
	std::string sparseNetworkDirectoryDvr_;
	std::vector<AdaptiveReconstructionMethodPtr> sparseNetworksIso_;
	std::vector<AdaptiveReconstructionMethodPtr> sparseNetworksDvr_;
	int selectedSparseNetworkIso_ = 0;
	int selectedSparseNetworkDvr_ = 0;
	bool samplingForceFlowToZero_ = false;
	//timings (in seconds)
	static constexpr int MAX_TIME_ENTRIES = 10;
	std::deque<float> timeRenderingLow_;
	std::deque<float> timeRenderingSamples_;
	std::deque<float> timeImportance_;
	std::deque<float> timeReconstruction_;

	//ADAPTIVE STEPSIZE
	std::string stepsizeImportanceDirectoryDvr_;
	std::vector<ImportanceSamplingMethodPtr> stepsizeImportanceSamplerDvr_;
	int selectedStepsizeImportanceSamplerDvr_ = 0;
	std::string stepsizeReconstructionDirectoryDvr_;
	std::vector<AdaptiveStepsizeReconstructionMethodPtr> stepsizeReconstructionDvr_;
	int selectedStepsizeReconstructionDvr_ = 0;
	float averageSamplesPerVoxel_ = 0;
	float minAdaptiveStepsize_ = 0;
	float maxAdaptiveStepsize_ = 0;
	
	//foveated
	bool foveatedEnable_ = false;
	int fovatedBlurRadiusPercent_ = 50;
	static const char* FoveatedBlurShapeNames[kernel::_FoveatedBlurShapeCount_];
	kernel::FoveatedBlurShape foveatedBlurShape_ = kernel::FoveatedBlurShapeSmoothstep;
	struct FoveatedLayerDesc
	{
		std::vector<SuperresolutionMethodPtr> possibleMethods_;
		SuperresolutionMethodPtr method_;
		int radius_ = 0; //percent of the diagonal
	};
	std::vector<FoveatedLayerDesc> foveatedLayers_;
	bool foveatedShowMasks_ = false;
	int2 foveatedCenter_ = int2{ 2,2 };
	bool foveatedLockMouse_ = false;

	//shading
	float3 ambientLightColor{ 0.1, 0.1, 0.1 };
	float3 diffuseLightColor{ 0.8, 0.8, 0.8 };
	float3 specularLightColor{ 0.1, 0.1, 0.1 };
	float specularExponent = 16;
	float3 materialColor{ 1.0, 1.0, 1.0 };
	float aoStrength = 0.5;
	float3 lightDirectionScreen{ 0,0,+1 };
	enum ChannelMode
	{
		ChannelMask,
		ChannelNormal,
		ChannelDepth,
		ChannelAO,
		ChannelFlow,
		ChannelColor,

		_ChannelCount_
	};
	static const char* ChannelModeNames[_ChannelCount_];
	ChannelMode channelMode_ = ChannelNormal;
	int temporalPostSmoothingPercentage_ = 0;
	bool flowWithInpainting_ = true;

	//screenshot
	std::string screenshotString_;
	float screenshotTimer_ = 0;

	//settings
	std::string settingsDirectory_;
	enum SettingsToLoad
	{
		CAMERA = 1<<0,
		COMPUTATION_MODE = 1<<1,
		TF_EDITOR = 1<<2,
		RENDERER = 1<<3,
		SHADING = 1<<4,
		DATASET = 1<<5,
		NETWORKS = 1<<6,
		_ALL_SETTINGS_ = 0xffffffff
	};
	int settingsToLoad_ = _ALL_SETTINGS_;

private:
	void releaseResources();
	
	void settingsSave();
	void settingsLoad();

	void selectMipmapLevel(int level, renderer::Volume::MipmapFilterMode filter, bool background = true);
	void loadVolume(const std::string& filename = "", bool background = true);
	void loadNetwork(RenderMode mode);
	void loadSamplingPattern();
	void loadSparseNetwork(RenderMode renderMode);
	void loadSamplingSequence(const std::string& filename = "");
	void loadImportanceNetwork(RenderMode renderMode, 
		std::string& directory, std::vector<ImportanceSamplingMethodPtr>& networks);
	void loadStepsizeReconstructionNetwork(RenderMode renderMode);
	
	void uiMenuBar();
	void uiVolume();
	void uiCamera();
	void uiRenderer();
	void uiTfEditor();
	void uiComputationMode();
	void uiSuperResolution(RenderMode renderMode);
	void uiAdaptiveSampling(RenderMode renderMode);
	void uiAdaptiveStepsize(RenderMode renderMode);
	void uiFoveated();
	void uiFoveatedMouse();
	void uiShading();
	void uiScreenshotOverlay();
	void uiFPSOverlay();
	void uiLockMouseOverlay();

	renderer::RendererArgs setupRendererArgs(
		RenderMode renderMode, int upscaleFactor=1);
	
	/**
	 * MAIN RENDERING: Renders the image using adaptive sampling.
	 */
	void renderAdaptiveSamples(RenderMode renderMode);
	static constexpr int RenderAdaptiveSamplesNoData = 0;
	static constexpr int RenderAdaptiveSamplesCanReconstruct = 1;
	static constexpr int RenderAdaptiveSamplesAlreadyFilled = 2;
	static constexpr int RenderAdaptiveSamplesNeedsSampling = 3;
	int renderAdaptiveSamples_SampleDynamic(RenderMode renderMode, ImportanceSamplingMethodPtr importanceSampler, bool clampAtOne);
	//Takes the normalizedImportanceMap_ from renderAdaptiveSamples_SampleDynamic and
	//computes the rendererSparseOutput_ and rendererInterpolatedOutput_
	int renderAdaptiveSamples_RenderSamples(RenderMode renderMode, bool requiresAO);
	int renderAdaptiveSamples_SampleFixed(RenderMode renderMode, bool requiresAO);
	void renderAdaptiveSamples_Reconstruct(RenderMode renderMode, AdaptiveReconstructionMethodPtr selectedNetwork);

	/**
	 * MAIN RENDERING: Renders the image using adaptive stepsize
	 */
	void renderAdaptiveStepsize(RenderMode renderMode);
	int renderAdaptiveStepsize_RenderSamples(RenderMode renderMode, bool requiresAO);
	void renderAdaptiveStepsize_Reconstruct(RenderMode renderMode, AdaptiveStepsizeReconstructionMethodPtr selectedNetwork);
	
	/**
	 * MAIN RENDERING: renders a fixed super-resolution.
	 */
	void renderSuperresolutionIso();
	/**
	 * MAIN RENDERING: renders a fixed super-resolution with DVR.
	 * Can be integrated into renderSuperresolution() in the future.
	 * However, for now it is best to keep them separated.
	 */
	void renderSuperresolutionDvr();
	/**
	 * Renders the foveated layers of higher resolution
	 * on top of screenTextureCudaBuffer_.
	 * Called within renderSuperresolution()
	 *
	 * Depends on the render arguments, channel selection
	 * and layer specification.
	 * Modifies screenTextureCudaBuffer_ in-place.
	 */
	void renderFoveated();

	void copyBufferToOpenGL();
	void resize(int display_w, int display_h);
	void triggerRedraw(RedrawMode mode);

	//Selects the channel to write to the cuda buffer.
	//The network output has shape (1 x Channels=8 x displayHeight_ x displayWidth_)
	//See renderer::IsoRendererOutputChannels
	void selectChannelIso(ChannelMode mode,
		const torch::Tensor& networkOutput,
		GLubyte* cudaBuffer) const;
	//Selects the channel to write to the cuda buffer.
	//The network output has shape (1 x Channels=10 x displayHeight_ x displayWidth_)
	//See renderer::DvrRendererOutputChannels
	void selectChannelDvr(ChannelMode mode,
		const torch::Tensor& networkOutput,
		GLubyte* cudaBuffer) const;
	
	void drawFoveatedMask(GLubyte* cudaBuffer) const;

	/**
	 * \brief Computes and prints statistics on how many samples
	 * are actually taken in the foveated rendering.
	 * This can estimate the time saving due to foveated rendering,
	 * assuming network execution costs nothing.
	 * \return percentage of samples of the full screen image used.
	 */
	float foveatedComputeNumberOfSamples();

	void screenshot();
};

