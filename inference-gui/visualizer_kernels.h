#pragma once

#include <cuda_runtime.h>
#include <torch/types.h>
#include <GL/glew.h>

#include "mesh_drawer.h"
#include <settings.h>
#include <volume.h>

namespace renderer {
	struct ShadingSettings;
}

namespace kernel
{
	/**
	 * \brief selects the output channel.
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param outputBuffer the output texture, RGBA of uint8
	 * \param r the channel used for the red output
	 * \param g the channel used for the green output
	 * \param b the channel used for the blue output
	 * \param a the channel used for the alpha output
	 * \param scaleRGB scaling used on the RGB colors: output = in*scale + offset
	 * \param offsetRGB offset used on the RGB colors: output = in*scale + offset
	 * \param scaleA scaling used on the alpha color: output = in*scale + offset
	 * \param offsetA offset used on the alpha color: output = in*scale + offset
	 */
	void selectOutputChannel(
		const torch::Tensor& inputTensor,
		GLubyte* outputBuffer,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB,
		float scaleA, float offsetA);

	//void selectOutputChannelSamples(
	//	const torch::Tensor& samplePositions,
	//	const torch::Tensor& sampleData,
	//	MeshDrawer::Vertex* outputVertices,
	//	int r, int g, int b, int a,
	//	float scaleRGB, float offsetRGB,
	//	float scaleA, float offsetA);

	/**
	 * \brief Performs the screen space shading.
	 * It assumes a tensor with the first 6 channels layouted as:
	 * mask, normalX, normalY, normalZ, depth, ao.
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param outputBuffer the output texture, RGBA of uint8
	 * \param settings
	 */
	void screenShading(
		const torch::Tensor& inputTensor,
		GLubyte* outputBuffer,
		const renderer::ShadingSettings& settings);

	enum FoveatedBlurShape
	{
		FoveatedBlurShapeLinear,
		FoveatedBlurShapeSmoothstep,
		_FoveatedBlurShapeCount_
	};
	struct LayerData
	{
		//subimage, RGBA of size width*height
		//can be NULL for color visualization
		GLubyte* subimage;
		int4 viewport; //minX, minY, width, height
		float smoothingRadius; //fraction of the subimage oval
		FoveatedBlurShape blurShape;
	};
	/**
	 * \brief Performs the foveated blending.
	 * The layers are blended in order, hence the top-most layer
	 * should be last in the vector of layers
	 * \param width the width of the screen
	 * \param height the height of the screen
	 * \param inoutBuffer the fullscreen input and output buffer
	 * \param layers the list of layers
	 */
	void foveatedBlending(
		int width, int height,
		GLubyte* inoutBuffer,
		const std::vector<LayerData>& layers);

	/**
	 * \brief Post-Processes the tensor after the network
	 * The tensor has a shape of 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU with the six channels
	 *  mask, normalX, normalY, normalZ, depth, ao.
	 * It transforms the mask from [-1,+1] to [0,1], normalizes the normal 
	 * and clamps mask, depth and ao.
	 */
	void networkPostProcessingUnshaded(torch::Tensor& inout);

	/**
	 * \brief Post-Processes the tensor after the adaptive sampling network
	 * The tensor has a shape of 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU with the eight channels
	 *  mask, normalX, normalY, normalZ, depth, ao, flowX, flowY.
	 * It transforms the mask from [-1,+1] to [0,1], normalizes the normal
	 * and clamps mask, depth and ao to [0,1], and flowX, flowY to [-1,1].
	 */
	void networkPostProcessingAdaptive(torch::Tensor& inout);

	/**
	 * \brief Performs nearest-neighbor upscaling of the input tensor
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param width the target width
	 * \param height the target height
	 * \return the output tensor
	 */
	torch::Tensor interpolateNearest(
		const torch::Tensor& inputTensor,
		int width, int height);

	/**
	 * \brief Performs nearest-neighbor upscaling of the input tensor
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param width the target width
	 * \param height the target height
	 * \return the output tensor
	 */
	torch::Tensor interpolateLinear(
		const torch::Tensor& inputTensor,
		int width, int height);

	/**
	 * \brief Performs nearest-neighbor upscaling of the input tensor
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param width the target width
	 * \param height the target height
	 * \return the output tensor
	 */
	torch::Tensor interpolateCubic(
		const torch::Tensor& inputTensor,
		int width, int height);

	enum InitialImage
	{
		InitialImageZero,
		InitialImageUnshaded,
		InitialImageInput
	};
	torch::Tensor createInitialImage(
		const torch::Tensor& currentInput,
		int channels,
		InitialImage mode,
		int width, int height);

	/**
	 * \brief Performs fast impainting
	 * \param mask the mask tensor of shape 1 * 1 * Height * Width.
	 *  Entries with value <=0 are considered "empty".
	 * \param data the tensor of shape 1 * Channel * Height * Width
	 *  that is impainted
	 * \return the impainted tensor of shape 1 * Channel * Height * Width
	 */
	torch::Tensor inpaintFlow(const torch::Tensor& mask, const torch::Tensor& data);

	/**
	 * \brief  Warps the high resolution input image with shape
          B x C x H*upscale_factor x W*upscale_factor
        with the upscaled low resolution flow in screen space with shape
          B x 2 x H x W.
        Output is the high resolution warped image.
	 * \param input_high 
	 * \param flow_low 
	 * \param upscale_factor 
	 * \return the warped input image
	 */
	torch::Tensor warpUpscale(
		const torch::Tensor& input_high,
		const torch::Tensor& flow_low,
		int upscale_factor);

	/**
	 * \brief The PyTorch libraries are initialized lazily,
	 * this method enforces the initialization.
	 */
	void initializePyTorch();


	/**
	 * \brief  Fills the color map using tfTexture which is created
		according to control points. Filled color map is then displayed
		in TF Editor menu.
	 * \param colorMap surface object which makes it possible to modify color map via surface writes.
	 * \param tfTexture
	 * \param width width of color map
	 * \param height height of color map
	 */
	void fillColorMap(cudaSurfaceObject_t colorMap, cudaTextureObject_t tfTexture, int width, int height);


	/**
	 * \brief  Simply transfers float values of output tensor to byte format.
	 * \param dvrOutput tensor output of DvrKernel
	 * \param byteOutput final image to be displayed on screen
	 * \param width width of dvrOutput
	 * \param height height of dvrOutput
	 */
	void transferDvrOutput(const at::Tensor& dvrOutput, GLubyte* byteOutput, int width, int height);
}
