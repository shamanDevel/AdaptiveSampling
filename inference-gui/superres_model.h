#pragma once

#include <torch/script.h>
#include "visualizer_kernels.h"
#include "visualizer_commons.h"

class SuperresolutionMethod
{
public:
	virtual ~SuperresolutionMethod() {}
	/**
	 * Returns the name of the method as displayed in the GUI
	 */
	virtual const std::string& name() const = 0;
	/**
	 * Returns the upscale factor
	 */
	virtual int upscaleFactor() const = 0;
	/**
	 * Returns true if the method requires AO on the input.
	 * False, if it does not need to be computed and can be set to 1.
	 */
	virtual bool requiresAO() const = 0;
	/**
	 * Returns true if the method requires a warped previous input.
	 * False, if it does not need to be computed and can be set empty.
	 */
	virtual bool requiresPrevious() const = 0;
	/**
	 * Returns true iff the method supports arbitrary upscale factors.
	 */
	virtual bool supportsArbitraryUpscaleFactors() const = 0;
	/**
	 * \brief Evaluates the superresolution method.
	 * \param renderingInput the input tensor of shape 1*B*H*W with eight channels
	 * (mask in [0,1], normal-x, normal-y, normal-z, depth, ao, flow-x, flow-y)
	 * \param previousOutput optionally, the previous, high resolution output
	 * of the same specs as the returned tensor
	 * \param upscaleFactor the requested super-resolution factor or
	 *   0 if supportsArbitrarySuperresolutionFactors() returned false.
	 * \return the output tensor of shape 1*B*H*W with
	 * width and height being \ref upscaleFactor() times larger than the input tensor
	 * and the output tensor having six channels
	 * (mask in [0,1], normal-x, normal-y, normal-z, depth, ao)
	 */
	virtual torch::Tensor eval(
		const torch::Tensor& renderingInput,
		const torch::Tensor& previousOutput,
		int upscaleFactor) = 0;

	virtual std::string filename() const { return ""; }

	//Parameter set by the GUI
	bool canDelete = true;
};
typedef std::shared_ptr<SuperresolutionMethod> SuperresolutionMethodPtr;

class BaselineSuperresolution : public SuperresolutionMethod
{
public:
	enum InterpolationMode
	{
		InterpolationNearest,
		InterpolationLinear,
		InterpolationBicubic,
		_InterpolationCount_
	};

private:
	struct this_is_private {};
public:
	BaselineSuperresolution(const this_is_private&, 
		InterpolationMode mode, int upscaleFactor, int outputChannels);

public:
	const std::string& name() const override { return name_; }
	int upscaleFactor() const override { return upscaleFactor_; }
	bool requiresAO() const override { return true; }
	bool requiresPrevious() const override { return false; }
	bool supportsArbitraryUpscaleFactors() const override { return upscaleFactor_ > 1; }
	torch::Tensor eval(
		const torch::Tensor& renderingInput, 
		const torch::Tensor& previousOutput,
		int upscaleFactor) override;

	static SuperresolutionMethodPtr GetMethod(
		InterpolationMode mode, int upscaleFactor, int outputChannels);

private:
	static const std::string InterpolationModeNames[_InterpolationCount_];
	const InterpolationMode interpolationMode_;
	const int upscaleFactor_;
	const std::string name_;
	const int outputChannels_;

	typedef std::tuple<InterpolationMode, int, int> MapKey_t;
	static std::map<MapKey_t, SuperresolutionMethodPtr> method_map;
};

class LoadedSuperresolutionModelIso : public SuperresolutionMethod
{
public:
	LoadedSuperresolutionModelIso(const std::string& filename);

	const std::string& name() const override;
	int upscaleFactor() const override;
	bool requiresAO() const override { return false; }
	bool requiresPrevious() const override { return true; }
	bool supportsArbitraryUpscaleFactors() const override { return networkUpscaleFactor_>1; }
	std::string filename() const override { return networkFilename_; }
	
	torch::Tensor eval(
		const torch::Tensor& renderingInput,
		const torch::Tensor& previousOutput,
		int upscaleFactor) override;

private:
	std::string networkFilename_;
	std::string name_;
	torch::jit::script::Module network_;
	torch::jit::script::Module network16_;
	int networkUpscaleFactor_;
	kernel::InitialImage networkInitialImage_;

	torch::Tensor callNetwork(
		const torch::Tensor& renderingInput,
		const torch::Tensor& previousOutput);
};

class LoadedSuperresolutionModelDvr : public SuperresolutionMethod
{
public:
	LoadedSuperresolutionModelDvr(const std::string& filename);

	const std::string& name() const override;
	int upscaleFactor() const override;
	bool requiresAO() const override { return false; }
	bool requiresPrevious() const override { return false; }
	bool supportsArbitraryUpscaleFactors() const override { return networkUpscaleFactor_ > 1; }
	std::string filename() const override { return networkFilename_; }
	
	torch::Tensor eval(
		const torch::Tensor& renderingInput,
		const torch::Tensor& previousOutput,
		int upscaleFactor) override;

private:
	std::string networkFilename_;
	std::string name_;
	torch::jit::script::Module network_;
	int networkUpscaleFactor_;
	kernel::InitialImage networkInitialImage_;
	int inputChannels_;
	int receivesNormal_;
	int receivesDepth_;

	torch::Tensor callNetwork(
		const torch::Tensor& renderingInput,
		const torch::Tensor& previousOutput);
};