#include "adaptive_model.h"
#include <sstream>
#include <json.hpp>
#include <filesystem>
#include "tinyformat.h"
#include "visualizer_kernels.h"
#include "hsv_normalization.h"

torch::Tensor AdaptiveReconstructionGroundTruth::eval(
	const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask,
	const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput)
{
	throw std::exception("should not be called, this is the ground truth");
}

bool AdaptiveReconstructionGroundTruth::isGroundTruth() const
{
	return true;
}

torch::Tensor AdaptiveReconstructionMask::eval(
	const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask, 
	const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput)
{
	auto inputCopy = inputSparse.clone();
	if (renderMode_ == IsosurfaceRendering)
		inputCopy.narrow(1,0,1) = inputSparseMask; //mask
	else
		inputCopy.narrow(1, 3, 1) = inputSparseMask; //alpha
	//inputCopy[0] = inputCopy[0].clamp(0);
	return inputCopy;
}

torch::Tensor AdaptiveReconstructionInterpolated::eval(
	const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask, 
	const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput)
{
	return inputInterpolated;
}

AdaptiveReconstructionModel::AdaptiveReconstructionModel(const std::string& filename, RenderMode renderMode)
	: renderMode_(renderMode)
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

	//test if we are in the case of fixed super-resolution networks
	//that are misused as reconstruction models
	std::vector<int64_t> inputChannels;
	if (settingsJson.contains("upscale_factor"))
	{
		if (settingsJson.at("upscale_factor").get<int>() != 1)
		{
			throw std::exception(
				"A fixed super-resolution network was "
				"selected as a reconstruction network, "
				"but then the upscale factor has to be 1.");
		}
		interpolateInput_ = true;
		residual_ = true;
		hardInput_ = false;
		externalFlow_ = true;
		expectMask_ = false;
		appendMask_ = false;

		if (renderMode == IsosurfaceRendering)
			for (int i = 0; i < 5; ++i) inputChannels.push_back(i); //first 5 channels only
		else
		{
			//DVR
			for (int i = 0; i < 4; ++i) inputChannels.push_back(i); //rgba
			if (settingsJson.contains("receives_normal") &&
				settingsJson.at("receives_normal").get<bool>())
				for (int i = 4; i < 7; ++i) inputChannels.push_back(i); //normal xyz
			if (settingsJson.contains("receives_depth") &&
				settingsJson.at("receives_depth").get<bool>())
				inputChannels.push_back(7); //depth
		}

		infoText_ = "Superresolution network as reconstruction net";
	}
	else
	{
		//regular reconstruction network
		interpolateInput_ = settingsJson["interpolateInput"].get<bool>();
		residual_ = settingsJson["residual"].get<bool>();
		hardInput_ = settingsJson["hardInput"].get<bool>();
		externalFlow_ = settingsJson.contains("externalFlow") ?
            settingsJson["externalFlow"].get<bool>() : false;
		expectMask_ = settingsJson.contains("expectMask") ?
            settingsJson["expectMask"].get<bool>() : true;

		std::stringstream s;
		if (settingsJson["architecture"].get<std::string>() == "UNet")
		{
			s << "UNet: depth=" << settingsJson["depth"].get<int>() <<
                ", filters=" << settingsJson["filters"].get<int>() <<
                ", padding=" << settingsJson["padding"].get<std::string>() << "\n";
		}
		s << "Interpolate input: " << (interpolateInput_ ? "yes" : "no") << "\n";
		s << "Residual connection: " << (residual_ ? "yes" : "no") << "\n";
		s << "Hard input: " << (hardInput_ ? "yes" : "no") << "\n";
		s << "External flow: " << (externalFlow_ ? "yes" : "no") << "\n";

		if (settingsJson.contains("input_channels") && settingsJson["input_channels"].is_array())
		{
			s << "Channels:";
			for (const auto& e : settingsJson["input_channels"])
			{
				std::string channelName = e[0].get<std::string>();
				int channelIndex = e[1].get<int>();
				s << " " << channelName;
				inputChannels.push_back(channelIndex);
			}
			if (renderMode == IsosurfaceRendering)
				inputChannels.push_back(5); //append AO
			appendMask_ = true; //sampling mask will be also added
		} else
		{
			if (renderMode != IsosurfaceRendering)
				throw std::exception("input_channels list required for non Iso-networks");
			for (int i = 0; i < 7; ++i) inputChannels.push_back(i); //first 7 channels only
			s << "Channels: mask normalX normalY normalZ depth";
			appendMask_ = false;
		}
		
		infoText_ = s.str();
	}
	normalizeHsvValue_ = false; // renderMode == DirectVolumeRendering;
	inputChannels_ = at::from_blob(inputChannels.data(), { int64_t(inputChannels.size()) },
		torch::dtype(torch::kLong).device(torch::kCPU)).clone().cuda();
	
	name_ = std::filesystem::path(filename).stem().string();
}

const std::string& AdaptiveReconstructionModel::name() const
{
	return name_;
}

const std::string& AdaptiveReconstructionModel::infoString() const
{
	return infoText_;
}

torch::Tensor AdaptiveReconstructionModel::eval(
	const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask,
	const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput)
{
	std::cout << "Sizes: sparse=" << inputSparse.sizes() << ", interp=" << inputInterpolated.sizes() <<
		", prevOut=" << previousOutput.sizes() << std::endl;
	
	try {
		if (inputSparse.size(0) != 1) throw std::exception(
			tfm::format("batch size must be one, but is %d", inputSparse.size(0)).c_str());
		const int C = inputSparse.size(1);
		if (inputSparse.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");

		int width = inputSparse.size(3);
		int height = inputSparse.size(2);

		auto mask = inputSparseMask; //torch::abs(inputSparse.slice(1, 0, 1, 1));

		float hsvScaling = 1;
		
		//preprocess, assemble input
		torch::Tensor input;
		if (renderMode_ == IsosurfaceRendering)
		{
			if (externalFlow_)
			{
				//flow is interpolated from the sparse samples, not passed to the network
				input = torch::cat({
					inputInterpolated.slice(1, 0, 6, 1),
					inputSparse.slice(1, 0, 1, 1) }, 1);
			}
			else
			{
				//flow is passed to the network
				input = torch::cat({
					inputInterpolated,
					inputSparse.slice(1, 0, 1, 1) }, 1);
			}

			//select channels
			if (!interpolateInput_)
				input = input * mask;
			input = input.index_select(1, inputChannels_);
			if (appendMask_)
				input = torch::cat({ input, mask }, 1);

			//append previous output
			if (externalFlow_)
				input = torch::cat({ input, previousOutput.slice(1, 0, 6, 1) }, 1);
			else
				input = torch::cat({ input, previousOutput }, 1);
		}
		else
		{
			//select channels and normalize
			input = inputInterpolated;
			if (normalizeHsvValue_)
			{
				int dim = input.dim() - 3;
				input.narrow(dim, 0, 3) = HsvNormalization::Instance()->
                    normalizeHsvValue(input.narrow(dim, 0, 3), hsvScaling);
			}
			if (!interpolateInput_)
				input = input * mask;
			input = input.index_select(1, inputChannels_);

			//append samples + rgba output
			if (appendMask_)
				input = torch::cat({ 
					input,
					mask,
					previousOutput.narrow(1, 0, 4) }, 1);
			else
				input = torch::cat({
					input,
					previousOutput.narrow(1, 0, 4) }, 1);
		}
		
		//run network, it returns a tuple (output, masks)
		std::cout << "Network input: (" << input.sizes() << ", " << mask.sizes() << ")" << std::endl;
		
		c10::IValue output;
		at::Tensor outputTensor;
		if (VisualizerEvaluateNetworksInHalfPrecision)
		{
			if (expectMask_)
				output = network16_.forward({ input.to(at::ScalarType::Half), mask.to(at::ScalarType::Half) });
			else
				output = network16_.forward({ input.to(at::ScalarType::Half) });

			if (output.isTuple())
				outputTensor = output.toTuple()->elements()[0].toTensor();
			else
				outputTensor = output.toTensor();

			outputTensor = outputTensor.to(at::ScalarType::Float);
		}
		else
		{
			if (expectMask_)
				output = network_.forward({ input, mask });
			else
				output = network_.forward({ input });

			//extract output
			if (output.isTuple())
				outputTensor = output.toTuple()->elements()[0].toTensor();
			else
				outputTensor = output.toTensor();
		}
		if (outputTensor.dim() == 3)
			outputTensor = outputTensor.unsqueeze_(0);
		std::cout << "Network output: " << outputTensor.sizes() << std::endl;

		// postprocessing
		if (renderMode_ == IsosurfaceRendering)
		{
			if (externalFlow_) //append flow again
				outputTensor = torch::cat({
                    outputTensor,
                    inputInterpolated.slice(1, 6, 8, 1) }, 1);
			kernel::networkPostProcessingAdaptive(outputTensor);
		}
		else
		{
			if (normalizeHsvValue_)
			{
				int dim = outputTensor.dim() - 3;
				outputTensor.narrow(dim, 0, 3) = HsvNormalization::Instance()->
                    denormalizeHsvValue(outputTensor.narrow(dim, 0, 3), hsvScaling);
			}
			//append normal, depth and flow again,
			//the network only produces color
			outputTensor = torch::cat({
				outputTensor,
				inputInterpolated.narrow(1, 4, inputInterpolated.size(1) - 4) }, 1);
		}
		
		return outputTensor;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
		return torch::zeros_like(inputSparse);
	}
}
