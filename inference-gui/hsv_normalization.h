#pragma once

#include <torch/script.h>

class HsvNormalization
{
private:
	HsvNormalization();
public:
	~HsvNormalization() {}

	static const HsvNormalization* Instance();

	/**
	 * Converts an RGB tensor of shape (*, 3, H, W)
	 * to a HSV tensor of shape (*, 3, H, W)
	 */
	torch::Tensor rgb2hsv(const torch::Tensor& rgb) const;

	/**
	 * Converts an HSV tensor of shape (*, 3, H, W)
	 * to a RGB tensor of shape (*, 3, H, W)
	 */
	torch::Tensor hsv2rgb(const torch::Tensor& hsv) const;

	/**
	 * \brief Normalizes the given RGB input tensor to have a maximum
	 * HSV value of 1.
	 * Normalization happens over all batches (if there are any).
	 * To undo the normalization, call \ref denormalizeHsvValue()
	 *  and pass the 'scalingOut' parameter as its 'scalingIn' parameter.
	 * \param in the input RGB tensor of shape (*, 3, H, W)
	 * \param scalingOut the output scaling to undo the normalization
	 * \return the normalized RGB tensor of shape (*, 3, H, W)
	 */
	torch::Tensor normalizeHsvValue(const torch::Tensor& in, float& scalingOut) const;

	/**
	 * \brief Undos the normalization from \ref normalizeHsvValue.
	 * \param in the HSV-value normalized tensor of shape (*, 3, H, W)
	 * \param scalingIn the value of 'scalingOut' from normalizeHsvValue.
	 * \return the denormalized rgb tensor of shape (*, 3, H, W)
	 */
	torch::Tensor denormalizeHsvValue(const torch::Tensor& in, float scalingIn) const;

private:
	mutable torch::jit::script::Module rgb2hsv_;
	mutable torch::jit::script::Module hsv2rgb_;
};

