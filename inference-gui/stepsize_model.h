#pragma once

#include <torch/script.h>
#include "visualizer_commons.h"

class AdaptiveStepsizeReconstructionMethod
{
	const std::string EMPTY = "";
public:
	virtual ~AdaptiveStepsizeReconstructionMethod() {}

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
	 * Returns true if the method requires AO on the input.
	 * False, if it does not need to be computed and can be set to 1.
	 */
	virtual bool requiresAO() const = 0;
	/**
	 * Returns true if the method requires a warped previous input.
	 * False, if it does not need to be computed and can be set empty.
	 */
	virtual bool requiresPrevious() const = 0;

	virtual bool isGroundTruth() const { return false; }

	virtual std::string filename() const { return ""; }

	/**
	 * \brief Evaluates the reconstruction from the sparse samples to the
	 * dense output.
	 *
	 * The channels are:
	 * - mask: [0,1] for dense, {-1,0,+1} for sparse
	 * - normalX
	 * - normalY
	 * - normalZ
	 * - depth
	 * - ao
	 * - flowX
	 * - flowY
	 * 
	 * \param input input from the renderer, B*C*H*W
	 * \param previousOutput the previous output, B*7*H*W
	 * \return the dense output, B*8*H*W
	 */
	virtual torch::Tensor eval(
		const torch::Tensor& input,
		const torch::Tensor& previousOutput) = 0;

	//Parameter set by the GUI
	bool canDelete = true;
};
typedef std::shared_ptr<AdaptiveStepsizeReconstructionMethod> AdaptiveStepsizeReconstructionMethodPtr;

class AdaptiveStepsizeReconstructionGroundTruth : public AdaptiveStepsizeReconstructionMethod
{
	std::string name_ = "Ground Truth";
public:
	const std::string& name() const override { return name_; }
	torch::Tensor eval(
		const torch::Tensor& input, const torch::Tensor& previousOutput) override;
	bool isGroundTruth() const override;
	bool requiresAO() const override { return true; }
	bool requiresPrevious() const override { return false; }
};

class AdaptiveStepsizeReconstructionBaseline : public AdaptiveStepsizeReconstructionMethod
{
	std::string name_ = "Baseline";
	const RenderMode renderMode_;
public:
	AdaptiveStepsizeReconstructionBaseline(RenderMode renderMode) : renderMode_(renderMode) {}
	const std::string& name() const override { return name_; }
	torch::Tensor eval(
		const torch::Tensor& input, const torch::Tensor& previousOutput) override;
	bool requiresAO() const override { return true; }
	bool requiresPrevious() const override { return false; }
};

class AdaptiveStepsizeReconstructionModel : public AdaptiveStepsizeReconstructionMethod
{
public:
	AdaptiveStepsizeReconstructionModel(const std::string& filename, RenderMode renderMode);
	const std::string& name() const override;
	const std::string& infoString() const override;
	bool requiresAO() const override { return false; }
	bool requiresPrevious() const override { return true; }
	torch::Tensor eval(
		const torch::Tensor& input, const torch::Tensor& previousOutput) override;
	std::string filename() const override { return networkFilename_; }
private:
	std::string networkFilename_;
	std::string name_;
	torch::jit::script::Module network_;
	torch::jit::script::Module network16_;

	const RenderMode renderMode_;
	bool residual_;
	torch::Tensor inputChannels_;
	std::string infoText_;
	bool normalizeHsvValue_;
};