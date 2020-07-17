#pragma once

#include <torch/script.h>
#include "visualizer_kernels.h"
#include "visualizer_commons.h"

// This is the C++ version of the Python network.importance package.

class ImportanceSamplingMethod
{
	const std::string EMPTY = "";
public:
	virtual ~ImportanceSamplingMethod() {}

	/**
	 * Returns the name of the method as displayed in the GUI
	 */
	virtual const std::string& name() const = 0;
	/**
	 * Returns an information string with details about the method.
	 * Can contain newline characters.
	 */
	virtual const std::string& infoString() const { return EMPTY; }

	/**
	 * Returns true if the previous output is required as input.
	 * If yes, it is warped and passed to \ref eval(...),
	 * if not, it is omitted and saves some computation time.
	 */
	virtual bool requiresPrevious() const { return false; }

	virtual std::string filename() const { return ""; }
	
	/**
	 * \brief Estimates the importance map from a dense low-resolution rendering.
	 *
	 * The channels are:
	 * - mask: [0,1]
	 * - normalX
	 * - normalY
	 * - normalZ
	 * - depth
	 * - ao
	 * - flowX
	 * - flowY
	 *
	 * \param inputLow low resolution input, 8*H*W
	 * \param upscaling the upscaling factor
	 * \param previousOutput the previous output of shape 1*(H*upscaling)*(W*upscaling)
	 * \return the importance map in [0,1] of shape (H*upscaling)*(W*upscaling)
	 */
	virtual torch::Tensor eval(
		const torch::Tensor& inputLow,
		int upscaling,
		const torch::Tensor& previousOutput) = 0;

	/**
	 * \brief Normalizes the importanceMap of shape Height * Width
        to have a specified minimal value and mean value.
        The minimal value is simply added to the importanceMap.
	 * \param importanceMap the importance map of shape (HxW)
	 * \param min the minimal value after normalization
	 * \param mean the mean value after normalization
	 * \param pad the number of pixels cropped from the border before
         taking the mean. These border pixels will be filled with the minimal value
	 * \param clampAtOne clamp the importance map at one
	 * \return 
	 */
	static torch::Tensor normalize(
		const torch::Tensor& importanceMap,
		float min, float mean, int pad,
		bool clampAtOne);
	
	//Parameter set by the GUI
	bool canDelete = true;
};
typedef std::shared_ptr< ImportanceSamplingMethod> ImportanceSamplingMethodPtr;

class ImportanceSamplingConstant : public ImportanceSamplingMethod
{
	const std::string NAME = "Constant";
public:
	const std::string& name() const override { return NAME; }
	torch::Tensor eval(const torch::Tensor& inputLow, int upscaling, const torch::Tensor& previousOutput) override;
};

class ImportanceSamplingNormalGrad : public ImportanceSamplingMethod
{
	const std::string NAME = "Normal Gradient";
public:
	ImportanceSamplingNormalGrad(int channelStart, int channelLength);
	const std::string& name() const override { return NAME; }
	torch::Tensor eval(const torch::Tensor& inputLow, int upscaling, const torch::Tensor& previousOutput) override;
private:
	int channelStart_;
	int channelLength_;
};

class ImportanceSamplingNetwork : public ImportanceSamplingMethod
{
public:
	ImportanceSamplingNetwork(const std::string& filename, RenderMode renderMode);
	const std::string& name() const override { return name_; }
	torch::Tensor eval(const torch::Tensor& inputLow, int upscaling, const torch::Tensor& previousOutput) override;
	const std::string& infoString() const override { return infoText_; }
	bool requiresPrevious() const override { return requiresPrevious_; }
	std::string filename() const override { return networkFilename_; }
private:
	std::string networkFilename_;
	std::string name_;
	torch::jit::script::Module network_;
	torch::jit::script::Module network16_;
	std::string infoText_;

	torch::Tensor inputChannels_;
	int networkUpscale_;
	int postUpscale_;
	kernel::InitialImage initialImage_;
	bool disableTemporal_;
	bool requiresPrevious_;
	bool scaleMask_;
	bool normalizeHsvValue_;
};
