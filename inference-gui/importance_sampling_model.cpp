#include "importance_sampling_model.h"
#include <filesystem>
#include <json.hpp>

#include "hsv_normalization.h"

torch::Tensor ImportanceSamplingMethod::normalize(
	const torch::Tensor& importanceMap, 
	float min, float mean, int pad, bool clampAtOne)
{
	/*
	 * Python:
	    assert min<=mean
        m = torch.mean(
            importanceMap[:, pad:-pad-1, pad:-pad-1],
            dim=[1,2], keepdim=True)
        m = torch.clamp(m, min=1e-7)
        return min + importanceMap * ((mean-min)/m)
	 */
	mean = std::max(min, mean);
	torch::Tensor croppedMap;
	if (pad == 0)
		croppedMap = importanceMap;
	else
	{
		croppedMap = torch::constant_pad_nd(importanceMap, { -pad,-pad,-pad,-pad }, 0);
		//croppedMap = importanceMap.narrow(0, pad, importanceMap.size(0) - 2 * pad);
		//croppedMap = importanceMap.narrow(1, pad, importanceMap.size(1) - 2 * pad);
	}
	float m = torch::mean(croppedMap).item().toFloat();
	m = std::max(m, 1e-7f);
	torch::Tensor scaledMap = min + croppedMap * ((mean - min) / m);
	if (clampAtOne)
		scaledMap.clamp_max_(1.0f);
	if (pad > 0)
		scaledMap = torch::constant_pad_nd(scaledMap, { pad,pad,pad,pad }, min);
	return scaledMap;
}

torch::Tensor ImportanceSamplingConstant::eval(const torch::Tensor& inputLow, int upscaling,
	const torch::Tensor& previousOutput)
{
	int H = inputLow.size(1) * upscaling;
	int W = inputLow.size(2) * upscaling;
	return 0.5f * torch::ones({ H, W }, 
		at::dtype(c10::ScalarType::Float).device(inputLow.device()));
}

ImportanceSamplingNormalGrad::ImportanceSamplingNormalGrad(int channelStart, int channelLength)
	: channelStart_(channelStart)
	, channelLength_(channelLength)
{
}

torch::Tensor ImportanceSamplingNormalGrad::eval(const torch::Tensor& inputLow, int upscaling,
	const torch::Tensor& previousOutput)
{
	int H = inputLow.size(1);
	int W = inputLow.size(2);
	// derivative along X / H
	auto inputX = torch::replication_pad2d(inputLow, { 0,0,1,1 });
	auto gX = inputX.narrow(1, 2, H) - inputX.narrow(1, 0, H);
	// derivative along Y / W
	auto inputY = torch::replication_pad2d(inputLow, { 1,1,0,0 });
	auto gY = inputY.narrow(2, 2, W) - inputY.narrow(2, 0, W);
	//add gradients and sum channels (normals)
	auto g = torch::abs(gX) + torch::abs(gY);
	g = torch::sum(g.narrow(0, channelStart_, channelLength_), { 0 }, true);
	//scale
	return torch::upsample_bilinear2d(g.unsqueeze(0), 
		{ int(H * upscaling), int(W * upscaling) }, false)[0][0];
}

ImportanceSamplingNetwork::ImportanceSamplingNetwork(const std::string& filename, RenderMode renderMode)
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
	const auto settingsJson = nlohmann::json::parse(settingsStr);

	networkUpscale_ = settingsJson["networkUpscale"].get<int>();
	postUpscale_ = settingsJson["postUpscale"].get<int>();
	std::string initialImage = settingsJson["initial_image"].get<std::string>();
	disableTemporal_ = settingsJson["disableTemporal"].get<bool>();
	requiresPrevious_ = settingsJson.count("requiresPrevious") > 0 ?
		settingsJson["requiresPrevious"].get<bool>() : false;
	normalizeHsvValue_ = renderMode == DirectVolumeRendering;
	scaleMask_ = renderMode == IsosurfaceRendering;
	std::string outputLayer = settingsJson["outputLayer"].get<std::string>();

	if (initialImage == "zero")
		initialImage_ = kernel::InitialImageZero;
	else if (initialImage == "unshaded")
		initialImage_ = kernel::InitialImageUnshaded;
	else if (initialImage == "input")
		initialImage_ = kernel::InitialImageZero;
	else
	{
		std::string msg = "unknown initial image setting: " + initialImage;
		throw std::exception(msg.c_str());
	}

	std::stringstream s;
	s << "Upscale: network=" << networkUpscale_ << ", post=" << postUpscale_ << "\n";
	s << "Initial image: " << initialImage << "\n";
	s << "Supports temporal consistency: " << (disableTemporal_ ? "no" : "yes") << "\n";
	s << "Output layer: " << outputLayer;

	if (!settingsJson.contains("input_channels"))
		throw std::exception("'input_channels' has to be specified, either as array or integer");
	std::vector<int64_t> inputChannels;
	if (settingsJson["input_channels"].is_array())
	{
		s << "Channels:";
		for (const auto& e : settingsJson["input_channels"])
		{
			std::string channelName = e[0].get<std::string>();
			int channelIndex = e[1].get<int>();
			s << " " << channelName;
			inputChannels.push_back(channelIndex);
		}
	}
	else
	{
		int num_channels = settingsJson["input_channels"].get<int>();
		for (int i = 0; i < num_channels; ++i) inputChannels.push_back(i);
	}
	inputChannels_ = at::from_blob(inputChannels.data(), { int64_t(inputChannels.size()) },
		torch::dtype(torch::kLong).device(torch::kCPU)).clone().cuda();
	
	infoText_ = s.str();
	name_ = std::filesystem::path(filename).stem().string();
}

torch::Tensor ImportanceSamplingNetwork::eval(const torch::Tensor& inputLow, int upscaling,
	const torch::Tensor& previousOutput)
{
	int C = inputLow.size(0);
	int H = inputLow.size(1);
	int W = inputLow.size(2);
	if (inputLow.dim() != 3) throw std::exception("tensor dimension must be 3 (C*H*W)");

	int networkUp = networkUpscale_ * postUpscale_;
	
	try {
		torch::Tensor input = inputLow.index_select(0, inputChannels_);
		if (scaleMask_)
			input[0] = input[0] * 2 - 1; // convert mask from [0,1] to [-1,1]
		if (normalizeHsvValue_)
		{
			float scaling;
			input.narrow(0, 0, 3) = HsvNormalization::Instance()->
				normalizeHsvValue(input.narrow(0, 0, 3), scaling);
		}

		input = input.unsqueeze(0);
		
		torch::Tensor networkInput;
		if (requiresPrevious_)
		{
			torch::Tensor previous;
			if (!previousOutput.defined() || disableTemporal_)
			{
				previous = kernel::createInitialImage(
                    inputLow, 1, initialImage_, 
					W*networkUpscale_, H*networkUpscale_);
			}
			else
			{
				previous = kernel::interpolateLinear(
                    previousOutput.unsqueeze(0),
                    W * networkUpscale_, H * networkUpscale_);
			}

			torch::Tensor previousFlat = previous //TODO: optimize flattening
                .contiguous()
                .view({ 1, 1, H, networkUpscale_, W, networkUpscale_ })
                .permute({ 0,1,3,5,2,4 })
                .contiguous()
                .view({ 1, 1 * networkUpscale_ * networkUpscale_, H, W });
			
			networkInput = torch::cat({ input, previousFlat }, 1);
		}
		else
		{
			//old importance network, no temporal component
			networkInput = input;
		}

		//run network
		networkInput.set_requires_grad(false);
		networkInput = networkInput.detach();
		const int importanceBorder = 32;
		if (scaleMask_)
		{
			networkInput = torch::cat({
                torch::constant_pad_nd(
                    networkInput.narrow(1,0,1),
                    { importanceBorder, importanceBorder, importanceBorder, importanceBorder },
                    -1),
                torch::constant_pad_nd(
                    networkInput.narrow(1,1,networkInput.size(1) - 1),
                    { importanceBorder, importanceBorder, importanceBorder, importanceBorder },
                    0)
                }, 1);
		} else
		{
			networkInput = torch::constant_pad_nd(
				networkInput,
				{ importanceBorder, importanceBorder, importanceBorder, importanceBorder },
				0);
		}

		torch::Tensor networkOutput;
		if (VisualizerEvaluateNetworksInHalfPrecision) {
			networkInput = networkInput.to(at::ScalarType::Half);
			auto rawOutput = network16_.forward({ networkInput });
			networkOutput = rawOutput.toTensor();
			networkOutput = networkOutput.to(at::ScalarType::Float);
		}
		else
		{
			auto rawOutput = network_.forward({ networkInput });
			networkOutput = rawOutput.toTensor();
		}
		
		networkOutput = torch::constant_pad_nd(
			networkOutput,
			{ -importanceBorder * networkUpscale_, -importanceBorder * networkUpscale_,
				-importanceBorder * networkUpscale_, -importanceBorder * networkUpscale_ },
			0);
		
		//postprocess
		torch::Tensor scaledOutput = kernel::interpolateLinear(
			networkOutput,
			W * upscaling, H * upscaling);
		
		return scaledOutput[0][0];
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
		return 0.5f * torch::ones({ H*upscaling, W*upscaling },
			at::dtype(c10::ScalarType::Float).device(inputLow.device()));
	}
}
