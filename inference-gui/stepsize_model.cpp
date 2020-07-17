#include "stepsize_model.h"
#include <sstream>
#include <json.hpp>
#include <filesystem>
#include "tinyformat.h"
#include "visualizer_kernels.h"
#include "hsv_normalization.h"
#include <renderer.h>

torch::Tensor AdaptiveStepsizeReconstructionGroundTruth::eval(
	const torch::Tensor& input, const torch::Tensor& previousOutput)
{
	throw std::exception("should not be called, this is the ground truth");
}

bool AdaptiveStepsizeReconstructionGroundTruth::isGroundTruth() const
{
	return true;
}

torch::Tensor AdaptiveStepsizeReconstructionBaseline::eval(
	const torch::Tensor& input, const torch::Tensor& previousOutput)
{
	int channels;
	if (renderMode_ == IsosurfaceRendering)
		channels = renderer::IsoRendererOutputChannels;
	else
		channels = renderer::DvrRendererOutputChannels;
	return input.narrow(1, 0, channels);
}

AdaptiveStepsizeReconstructionModel::AdaptiveStepsizeReconstructionModel(const std::string& filename, RenderMode renderMode)
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
		residual_ = true;

		if (renderMode == IsosurfaceRendering)
			for (int i = 0; i < 5; ++i) inputChannels.push_back(i); //first 5 channels only
		else
		{
			//dvr, has 'input_channels' entry
			for (int i = 0; i < 4; ++i) inputChannels.push_back(i); //rgba
			if (settingsJson.at("receives_normal").get<bool>())
				for (int i = 4; i < 7; ++i) inputChannels.push_back(i); //normal xyz
			if (settingsJson.at("receives_normal").get<bool>())
				inputChannels.push_back(7); //depth
		}

		infoText_ = "Superresolution network as reconstruction net";
	}
	else
	{
		//regular reconstruction network
		residual_ = settingsJson["residual"].get<bool>();

		std::stringstream s;
		if (settingsJson["architecture"].get<std::string>() == "UNet")
		{
			s << "UNet: depth=" << settingsJson["depth"].get<int>() <<
                ", filters=" << settingsJson["filters"].get<int>() <<
                ", padding=" << settingsJson["padding"].get<std::string>() << "\n";
		} else if (settingsJson["architecture"].get<std::string>() == "UNet")
		{
			s << "EnhanceNet: depth=" << settingsJson["depth"].get<int>() <<
				", filters=" << settingsJson["filters"].get<int>() <<
				", padding=" << settingsJson["padding"].get<std::string>() << "\n";
		}
		s << "Residual connection: " << (residual_ ? "yes" : "no") << "\n";

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
		} else
		{
			if (renderMode != IsosurfaceRendering)
				throw std::exception("input_channels list required for non Iso-networks");
			for (int i = 0; i < 7; ++i) inputChannels.push_back(i); //first 7 channels only
			s << "Channels: mask normalX normalY normalZ depth";
		}
		
		infoText_ = s.str();
	}
	normalizeHsvValue_ = renderMode == DirectVolumeRendering;
	inputChannels_ = at::from_blob(inputChannels.data(), { int64_t(inputChannels.size()) },
		torch::dtype(torch::kLong).device(torch::kCPU)).clone().cuda();
	
	name_ = std::filesystem::path(filename).stem().string();
}

const std::string& AdaptiveStepsizeReconstructionModel::name() const
{
	return name_;
}

const std::string& AdaptiveStepsizeReconstructionModel::infoString() const
{
	return infoText_;
}

torch::Tensor AdaptiveStepsizeReconstructionModel::eval(
	const torch::Tensor& inputOriginal, const torch::Tensor& previousOutput)
{
	try {
		if (inputOriginal.size(0) != 1) throw std::exception(
			tfm::format("batch size must be one, but is %d", inputOriginal.size(0)).c_str());
		const int C = inputOriginal.size(1);
		if (inputOriginal.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");

		int width = inputOriginal.size(3);
		int height = inputOriginal.size(2);

		float hsvScaling = 1;
		
		//preprocess, assemble input
		torch::Tensor input = inputOriginal;
		if (renderMode_ == IsosurfaceRendering)
		{
			//select channels
			input = input.index_select(1, inputChannels_);

			//append previous output
			input = torch::cat({ input, previousOutput }, 1);
		}
		else
		{
			//select channels and normalize
			input = inputOriginal;
			if (normalizeHsvValue_)
			{
				int dim = input.dim() - 3;
				input.narrow(dim, 0, 3) = HsvNormalization::Instance()->
                    normalizeHsvValue(input.narrow(dim, 0, 3), hsvScaling);
			}
			input = input.index_select(1, inputChannels_);

			//append samples + rgba output
			input = torch::cat({ 
				input,
				previousOutput.narrow(1, 0, 4) }, 1);
		}
		
		//run network, it returns a tuple (output, masks)
		c10::IValue output;
		at::Tensor outputTensor;
		if (VisualizerEvaluateNetworksInHalfPrecision)
		{
			input = input.to(at::ScalarType::Half);
			output = network16_.forward({ input });
			if (output.isTuple())
				outputTensor = output.toTuple()->elements()[0].toTensor();
			else
				outputTensor = output.toTensor();
			outputTensor = outputTensor.to(at::ScalarType::Float);
		}
		else
		{
			output = network_.forward({ input });
			if (output.isTuple())
				outputTensor = output.toTuple()->elements()[0].toTensor();
			else
				outputTensor = output.toTensor();
		}
		std::cout << "Network output: " << outputTensor.sizes() << std::endl;

		// postprocessing
		if (renderMode_ == IsosurfaceRendering)
		{
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
				inputOriginal.narrow(1, 4, inputOriginal.size(1) - 4) }, 1);
		}
		
		return outputTensor;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
		return torch::zeros_like(inputOriginal);
	}
}
