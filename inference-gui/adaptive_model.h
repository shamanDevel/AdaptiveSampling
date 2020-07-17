#pragma once

#include <torch/script.h>
#include "visualizer_commons.h"

class AdaptiveReconstructionMethod
{
	const std::string EMPTY = "";
public:
	virtual ~AdaptiveReconstructionMethod() {}

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
	/**
	 * Specifies the source of the optical flow for warping.
	 * False: The model generates the flow from the current input.
	 *   Hence, the generated flow from the last frame is used to warp the last image.
	 * True: The interpolated flow from the current sparse samples before sending it
	 *   to the network is used to warp the last frame
	 */
	virtual bool externalFlow() const { return true; }

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
	 * \param inputSparse sparse input from the mesh drawer, B*8*H*W
	 * \param inputSparseMask mask where the samples are defined (1) or not (0), B*1*H*W
	 * \param inputInterpolated interpolated input from the mesh drawer, B*8*H*W
	 * \param previousOutput the previous output, B*7*H*W
	 * \return the dense output, B*8*H*W
	 */
	virtual torch::Tensor eval(
		const torch::Tensor& inputSparse,
		const torch::Tensor& inputSparseMask,
		const torch::Tensor& inputInterpolated,
		const torch::Tensor& previousOutput) = 0;

	//Parameter set by the GUI
	bool canDelete = true;
};
typedef std::shared_ptr<AdaptiveReconstructionMethod> AdaptiveReconstructionMethodPtr;

class AdaptiveReconstructionGroundTruth : public AdaptiveReconstructionMethod
{
	std::string name_ = "Ground Truth";
public:
	const std::string& name() const override { return name_; }
	torch::Tensor eval(
		const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask, 
		const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput) override;
	bool isGroundTruth() const override;
	bool requiresAO() const override { return true; }
	bool requiresPrevious() const override { return false; }
};

class AdaptiveReconstructionMask : public AdaptiveReconstructionMethod
{
	std::string name_ = "Only Samples";
	const RenderMode renderMode_;
public:
	AdaptiveReconstructionMask(RenderMode renderMode) : renderMode_(renderMode) {}
	const std::string& name() const override { return name_; }
	torch::Tensor eval(
		const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask, 
		const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput) override;
	bool requiresAO() const override { return true; }
	bool requiresPrevious() const override { return false; }
};

class AdaptiveReconstructionInterpolated : public AdaptiveReconstructionMethod
{
	std::string name_ = "Interpolated";
public:
	const std::string& name() const override { return name_; }
	torch::Tensor eval(
		const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask, 
		const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput) override;
	bool requiresAO() const override { return true; }
	bool requiresPrevious() const override { return false; }
};

class AdaptiveReconstructionModel : public AdaptiveReconstructionMethod
{
public:
	AdaptiveReconstructionModel(const std::string& filename, RenderMode renderMode);
	const std::string& name() const override;
	const std::string& infoString() const override;
	bool requiresAO() const override { return false; }
	bool requiresPrevious() const override { return true; }
	torch::Tensor eval(
		const torch::Tensor& inputSparse, const torch::Tensor& inputSparseMask, 
		const torch::Tensor& inputInterpolated, const torch::Tensor& previousOutput) override;
	bool externalFlow() const override { return externalFlow_; }
	std::string filename() const override { return networkFilename_; }
private:
	std::string networkFilename_;
	std::string name_;
	torch::jit::script::Module network_;
	torch::jit::script::Module network16_;

	const RenderMode renderMode_;
	bool interpolateInput_;
	bool residual_;
	bool hardInput_;
	bool externalFlow_;
	bool expectMask_;
	bool appendMask_;
	torch::Tensor inputChannels_;
	std::string infoText_;
	bool normalizeHsvValue_;
};