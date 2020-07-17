#include "superres_model.h"
#include <json.hpp>
#include <experimental/filesystem>
#include <cuMat/src/Context.h>
#include "helper_math.h"

const std::string BaselineSuperresolution::InterpolationModeNames[] = {
	"Nearest", "Bilinear", "Bicubic"
};

std::map<BaselineSuperresolution::MapKey_t, SuperresolutionMethodPtr> BaselineSuperresolution::method_map;

BaselineSuperresolution::BaselineSuperresolution(const this_is_private&, 
	InterpolationMode mode, int upscaleFactor, int outputChannels)
	: interpolationMode_(mode)
	, upscaleFactor_(upscaleFactor)
	, name_(upscaleFactor == 1 ? "Ground Truth" : (
		InterpolationModeNames[mode] + " (" + std::to_string(upscaleFactor) + "x)"
		))
	, outputChannels_(outputChannels)
{}

torch::Tensor BaselineSuperresolution::eval(const torch::Tensor& renderingInput,
	const torch::Tensor& previousOutput, int superresFactor)
{
	torch::Tensor input = renderingInput.narrow(1, 0, outputChannels_);
	if (superresFactor == 0)
		superresFactor = upscaleFactor_;
	int width = input.size(3) * superresFactor;
	int height = input.size(2) * superresFactor;
	switch (interpolationMode_)
	{
	case InterpolationNearest: return kernel::interpolateNearest(input, width, height);
	case InterpolationLinear: return kernel::interpolateLinear(input, width, height);
	case InterpolationBicubic: return kernel::interpolateCubic(input, width, height);
	default: throw std::exception("unknown enum");
	}
}

SuperresolutionMethodPtr BaselineSuperresolution::GetMethod(
	InterpolationMode mode, int upscaleFactor, int outputChannels)
{
	MapKey_t key{ mode, upscaleFactor, outputChannels };
	if (method_map.count(key) > 0)
		return method_map.at(key);
	SuperresolutionMethodPtr m = std::make_shared<BaselineSuperresolution>(
		this_is_private(), mode, upscaleFactor, outputChannels);
	method_map.emplace(key, m);
	return m;
}

LoadedSuperresolutionModelIso::LoadedSuperresolutionModelIso(const std::string& filename)
{
	networkFilename_ = filename;
	torch::jit::script::ExtraFilesMap extra_files;
	extra_files.emplace("settings.json", "");
	network_ = torch::jit::load(filename, c10::kCUDA, extra_files);
	network_.to(torch::kCUDA);
	network_.eval();
	network16_ = torch::jit::load(filename, c10::kCUDA, extra_files);
	network16_.to(torch::kCUDA, at::ScalarType::Half);
	network16_.eval();
	const std::string settingsStr = extra_files["settings.json"];
	auto settingsJson = nlohmann::json::parse(settingsStr);

	networkUpscaleFactor_ = settingsJson["upscale_factor"].get<int>();
	std::string initialImage = settingsJson["initial_image"].get<std::string>();
	int inputChannels = settingsJson["input_channels"].get<int>();

	if (initialImage == "zero")
		networkInitialImage_ = kernel::InitialImageZero;
	else if (initialImage == "unshaded")
		networkInitialImage_ = kernel::InitialImageUnshaded;
	else if (initialImage == "input")
		networkInitialImage_ = kernel::InitialImageZero;
	else
	{
		std::string msg = "unknown initial image setting: " + initialImage;
		throw std::exception(msg.c_str());
	}

	int expectedChannels = 5 + 6 * networkUpscaleFactor_ * networkUpscaleFactor_;
	if (inputChannels != expectedChannels)
	{
		std::string msg = "wrong channel count, expected 5 + 6 * upscale_factor^2 = "
			+ std::to_string(expectedChannels) + ", but got " + std::to_string(inputChannels);
		throw std::exception(msg.c_str());
	}

	name_ = std::experimental::filesystem::path(filename).stem().string() +
		" (" + std::to_string(networkUpscaleFactor_) + "x)";
}

const std::string& LoadedSuperresolutionModelIso::name() const
{
	return name_;
}

int LoadedSuperresolutionModelIso::upscaleFactor() const
{
	return networkUpscaleFactor_;
}

torch::Tensor LoadedSuperresolutionModelIso::callNetwork(
	const torch::Tensor& renderingInput,
	const torch::Tensor& previousOutput)
{
	const int width = renderingInput.size(3);
	const int height = renderingInput.size(2);
	const int widthHigh = width * networkUpscaleFactor_;
	const int heightHigh = height * networkUpscaleFactor_;

	// preprocessing (transform mask)
	torch::Tensor networkInput1 = renderingInput.narrow(1, 0, 5);
	networkInput1[0][0] = networkInput1[0][0] * 2 - 1;
	
	if (previousOutput.size(1) != 6) throw std::exception(
		("previous output must have 6 channels, but had "+std::to_string(previousOutput.size(1))).c_str());
	if (previousOutput.size(2) != heightHigh)
		throw std::exception(("previous output does not match expected size: " +
			std::to_string(previousOutput.size(2)) +
			" vs " +
			std::to_string(heightHigh)).c_str());
	if (previousOutput.size(3) != widthHigh)
		throw std::exception(("previous output does not match expected size: " +
			std::to_string(previousOutput.size(3)) +
			" vs " +
			std::to_string(widthHigh)).c_str());
	
	torch::Tensor previousFlat = previousOutput //TODO: optimize flattening
		.contiguous()
		.view({ 1, 6, height, networkUpscaleFactor_, width, networkUpscaleFactor_ })
		.permute({ 0,1,3,5,2,4 })
		.contiguous()
		.view({ 1, 6 * networkUpscaleFactor_ * networkUpscaleFactor_, height, width });
	std::cout << "networkInput1: " << networkInput1.sizes() << std::endl;
	std::cout << "previousFlat: " << previousFlat.sizes() << std::endl;
	torch::Tensor networkInput = torch::cat({ networkInput1, previousFlat }, 1);

	// network
	std::cout << "Network input: " << networkInput.sizes() << " : " <<
		networkInput.dtype() << " : " << networkInput.device() << std::endl;
	std::cout << "Network: " << " : " << network_.num_slots() << std::endl;
	networkInput.set_requires_grad(false);
	networkInput.detach_();

	torch::Tensor networkOutput;
	if (VisualizerEvaluateNetworksInHalfPrecision)
	{
		networkInput = networkInput.to(at::ScalarType::Half);
		auto rawOutput = network16_.forward({ networkInput });
		networkOutput = rawOutput.toTuple()->elements()[0].toTensor();
		networkOutput = networkOutput.to(at::ScalarType::Float);
	}
	else
	{
		auto rawOutput = network_.forward({ networkInput });
		networkOutput = rawOutput.toTuple()->elements()[0].toTensor();
	}
	
	networkOutput.set_requires_grad(false);
	networkOutput.detach_();

	// postprocessing
	kernel::networkPostProcessingUnshaded(networkOutput);
	return networkOutput;
}

torch::Tensor LoadedSuperresolutionModelIso::eval(
	const torch::Tensor& renderingInput,
	const torch::Tensor& previousOutput,
	const int upscaleFactor)
{
	if (renderingInput.size(0) != 1) throw std::exception("batch size must be one");
	if (renderingInput.size(1) < 5) throw std::exception("input must have at least 5 channels");
	if (renderingInput.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");
	
	const int width = renderingInput.size(3);
	const int height = renderingInput.size(2);
	const int widthHigh = width * upscaleFactor;
	const int heightHigh = height * upscaleFactor;

	const torch::Tensor previousOutput1 = previousOutput.defined()
		? previousOutput
		: kernel::createInitialImage(
			renderingInput, 6, networkInitialImage_, 
			widthHigh, heightHigh);

	bool postDownsample = true;
	int currentUpscaleFactor = 1;
	torch::Tensor currentInout = renderingInput;

	try {
		do
		{
			if (postDownsample)
			{
				//first upscale by the network, then bilinear downscale if too large
				torch::Tensor previous = kernel::interpolateLinear(
					previousOutput1,
					width*currentUpscaleFactor*networkUpscaleFactor_,
					height*currentUpscaleFactor*networkUpscaleFactor_);
				currentInout = callNetwork(currentInout, previous);
				currentUpscaleFactor *= networkUpscaleFactor_;
				if (currentUpscaleFactor > upscaleFactor)
				{ //we overshot, scale down
					currentInout = kernel::interpolateLinear(
						currentInout,
						width * upscaleFactor,
						height * upscaleFactor);
					currentUpscaleFactor = upscaleFactor;
				}
				postDownsample = false;
			}
			else
			{
				//prepare inout
				torch::Tensor previous;
				if (currentUpscaleFactor * networkUpscaleFactor_ > upscaleFactor)
				{
					//we would overshoot, so scale down first to the perfect match
					//this avoids very different feature scales and out-of-memory errors
					currentInout = kernel::interpolateLinear(
						currentInout,
						width * upscaleFactor / networkUpscaleFactor_,
						height * upscaleFactor / networkUpscaleFactor_);
					previous = kernel::interpolateLinear(
						previousOutput1,
						width * upscaleFactor / networkUpscaleFactor_ * networkUpscaleFactor_,
						height * upscaleFactor / networkUpscaleFactor_ * networkUpscaleFactor_);
					currentUpscaleFactor = upscaleFactor;
				}
				else
				{
					//we are still not there
					previous = kernel::interpolateLinear(
						previousOutput1,
						width*currentUpscaleFactor*networkUpscaleFactor_,
						height*currentUpscaleFactor*networkUpscaleFactor_);
					currentUpscaleFactor *= networkUpscaleFactor_;
				}
				//run network
				currentInout = callNetwork(currentInout, previous);
			}
		} while (currentUpscaleFactor < upscaleFactor);
		
		return currentInout;
	} catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
		return torch::zeros(
			{ 1, 6, heightHigh, widthHigh}, 
			torch::dtype(torch::kFloat).device(torch::kCUDA));
	}
}

LoadedSuperresolutionModelDvr::LoadedSuperresolutionModelDvr(const std::string& filename)
{
	networkFilename_ = filename;
	torch::jit::script::ExtraFilesMap extra_files;
	extra_files.emplace("settings.json", "");
	network_ = torch::jit::load(filename, c10::kCUDA, extra_files);
	network_.to(torch::kCUDA);
	network_.eval();
	const std::string settingsStr = extra_files["settings.json"];
	auto settingsJson = nlohmann::json::parse(settingsStr);

	networkUpscaleFactor_ = settingsJson["upscale_factor"].get<int>();
	std::string initialImage = settingsJson["initial_image"].get<std::string>();
	int inputChannels = settingsJson["input_channels"].get<int>();

	if (initialImage == "zero")
		networkInitialImage_ = kernel::InitialImageZero;
	else if (initialImage == "unshaded")
		networkInitialImage_ = kernel::InitialImageUnshaded;
	else if (initialImage == "input")
		networkInitialImage_ = kernel::InitialImageZero;
	else
	{
		std::string msg = "unknown initial image setting: " + initialImage;
		throw std::exception(msg.c_str());
	}

	inputChannels_ = inputChannels - 4 * networkUpscaleFactor_ * networkUpscaleFactor_;
	if (inputChannels_ < 4 || inputChannels_ > 8)
	{
		std::string msg = "input channel count has to be between 4 (rgba) and 8 (rgba,normal,depth), but got "
			+ std::to_string(inputChannels_);
		throw std::exception(msg.c_str());
	}
	receivesNormal_ = settingsJson.contains("receives_normal")
		? settingsJson["receives_normal"].get<bool>() : false;
	receivesDepth_ = settingsJson.contains("receives_depth")
		? settingsJson["receives_depth"].get<bool>() : false;

	name_ = std::experimental::filesystem::path(filename).stem().string() +
		" (" + std::to_string(networkUpscaleFactor_) + "x)";
}

const std::string & LoadedSuperresolutionModelDvr::name() const
{
	return name_;
}

int LoadedSuperresolutionModelDvr::upscaleFactor() const
{
	return networkUpscaleFactor_;
}

torch::Tensor LoadedSuperresolutionModelDvr::callNetwork(const torch::Tensor& renderingInput,
	const torch::Tensor& previousOutput)
{
	const int width = renderingInput.size(3);
	const int height = renderingInput.size(2);
	const int widthHigh = width * networkUpscaleFactor_;
	const int heightHigh = height * networkUpscaleFactor_;

	if (previousOutput.size(2) != heightHigh)
		throw std::exception(("previous output does not match expected size: " +
			std::to_string(previousOutput.size(2)) +
			" vs " +
			std::to_string(heightHigh)).c_str());
	if (previousOutput.size(3) != widthHigh)
		throw std::exception(("previous output does not match expected size: " +
			std::to_string(previousOutput.size(3)) +
			" vs " +
			std::to_string(widthHigh)).c_str());

	//flatten previous
	torch::Tensor previousFlat = previousOutput //TODO: optimize flattening
		.contiguous()
		.view({ 1, 4, height, networkUpscaleFactor_, width, networkUpscaleFactor_ })
		.permute({ 0,1,3,5,2,4 })
		.contiguous()
		.view({ 1, 4 * networkUpscaleFactor_ * networkUpscaleFactor_, height, width });

	//select channels from the input
	torch::Tensor renderingInput1;
	if (receivesNormal_ && receivesDepth_)
	{	//red, green, blue, alpha, normal x, y, z, depth.  No flow x,y
		renderingInput1 = renderingInput.narrow(1, 0, 8);
	}
	else if (receivesNormal_ && !receivesDepth_)
	{	//red, green, blue, alpha, normal x, y, z. No depth, flow x,y
		renderingInput1 = renderingInput.narrow(1, 0, 7);
	}
	else if (!receivesNormal_ && receivesDepth_)
	{	//red, green, blue, alpha, depth. No normal x, y, z, flow x,y
		renderingInput1 = torch::cat({
			renderingInput.narrow(1, 0, 4),
			renderingInput.narrow(1, 7, 1)
		}, 1);
	} else
	{
		//red, green, blue, alpha. No normal, depth, flow
		renderingInput1 = renderingInput.narrow(1, 0, 4);
	}
	
	std::cout << "renderingInput: " << renderingInput1.sizes() << std::endl;
	std::cout << "previousFlat: " << previousFlat.sizes() << std::endl;
	torch::Tensor networkInput = torch::cat({ renderingInput1, previousFlat }, 1);

	// network
	std::cout << "Network input: " << networkInput.sizes() << " : " <<
		networkInput.dtype() << " : " << networkInput.device() << std::endl;
	std::cout << "Network: " << " : " << network_.num_slots() << std::endl;
	networkInput.set_requires_grad(false);
	networkInput.detach_();
	auto rawOutput = network_.forward({ networkInput });
	torch::Tensor networkOutput = rawOutput.toTuple()->elements()[0].toTensor();
	networkOutput.set_requires_grad(false);
	networkOutput.detach_();

	return networkOutput;
}

torch::Tensor LoadedSuperresolutionModelDvr::eval(
	const torch::Tensor& renderingInput, const torch::Tensor& previousOutput,
	int upscaleFactor)
{
	if (renderingInput.size(0) != 1) throw std::exception("batch size must be one");
	if (renderingInput.size(1) < 4) throw std::exception("input must have at least 4 channels");
	if (renderingInput.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");

	const int width = renderingInput.size(3);
	const int height = renderingInput.size(2);
	const int widthHigh = width * upscaleFactor;
	const int heightHigh = height * upscaleFactor;

	const torch::Tensor previousOutput1 = previousOutput.defined()
		? previousOutput.narrow(1, 0, 4) //only color, no normal, depth
		: kernel::createInitialImage(
			renderingInput, 4, networkInitialImage_,
			widthHigh, heightHigh);

	bool postDownsample = true;
	int currentUpscaleFactor = 1;
	torch::Tensor currentInout = renderingInput;
	
	try
	{
		do {
			if (postDownsample)
			{
				//first upscale by the network, then bilinear downscale if too large
				torch::Tensor previous = kernel::interpolateLinear(
					previousOutput1,
					width*currentUpscaleFactor*networkUpscaleFactor_,
					height*currentUpscaleFactor*networkUpscaleFactor_);
				currentInout = callNetwork(currentInout, previous);
				currentUpscaleFactor *= networkUpscaleFactor_;
				if (currentUpscaleFactor > upscaleFactor)
				{ //we overshot, scale down
					currentInout = kernel::interpolateLinear(
						currentInout,
						width * upscaleFactor,
						height * upscaleFactor);
					currentUpscaleFactor = upscaleFactor;
				}
				postDownsample = false;
			}
			else
			{
				//prepare inout
				torch::Tensor previous;
				if (currentUpscaleFactor * networkUpscaleFactor_ > upscaleFactor)
				{
					//we would overshoot, so scale down first to the perfect match
					//this avoids very different feature scales and out-of-memory errors
					currentInout = kernel::interpolateLinear(
						currentInout,
						width * upscaleFactor / networkUpscaleFactor_,
						height * upscaleFactor / networkUpscaleFactor_);
					previous = kernel::interpolateLinear(
						previousOutput1,
						width * upscaleFactor / networkUpscaleFactor_ * networkUpscaleFactor_,
						height * upscaleFactor / networkUpscaleFactor_ * networkUpscaleFactor_);
					currentUpscaleFactor = upscaleFactor;
				}
				else
				{
					//we are still not there
					previous = kernel::interpolateLinear(
						previousOutput1,
						width*currentUpscaleFactor*networkUpscaleFactor_,
						height*currentUpscaleFactor*networkUpscaleFactor_);
					currentUpscaleFactor *= networkUpscaleFactor_;
				}
				//run network
				currentInout = callNetwork(currentInout, previous);
			}
		} while (currentUpscaleFactor < upscaleFactor);

		//append interpolated normal and depth
		auto normalDepthHigh = kernel::interpolateLinear(
			renderingInput.narrow(1, 4, 4),
			widthHigh, heightHigh);
		return torch::cat({
			currentInout,
			normalDepthHigh,
			}, 1);
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
		return torch::zeros({ 1, 4, heightHigh, widthHigh }, torch::dtype(torch::kFloat).device(torch::kCUDA));
	}
}
