#include "visualizer_kernels.h"
#include "helper_math.h"
#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <inpainting.h>

__device__ unsigned int rgbaToInt(float r, float g, float b, float a)
{
	r = clamp(r*255, 0.0f, 255.0f);
	g = clamp(g*255, 0.0f, 255.0f);
	b = clamp(b*255, 0.0f, 255.0f);
	a = clamp(a*255, 0.0f, 255.0f);
	return (unsigned int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
	//return 0xff000000 | (int(b) << 16) | (int(g) << 8) | int(r);
}
__device__ unsigned int float4ToInt(float4 rgba)
{
	return rgbaToInt(rgba.x, rgba.y, rgba.z, rgba.w);
}
__device__ float4 intToFloat4(unsigned int rgba)
{
	return make_float4(
		(rgba & 0xff) / 255.0f,
		((rgba >> 8) & 0xff) / 255.0f,
		((rgba >> 16) & 0xff) / 255.0f,
		((rgba >> 24) & 0xff) / 255.0f
	);
}

__device__ inline float fetchChannel(
	torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> input,
	int channel, int x, int y)
{
	if (channel == -1) return 0;
	if (channel == -2) return 1;
	return input[0][channel][y][x];
}
__global__ void SelectOutputChannelKernel(
	dim3 virtual_size,
	torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> input,
	unsigned int* output,
	int rId, int gId, int bId, int aId,
	float scaleRGB, float offsetRGB, float scaleA, float offsetA)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)

		float r = fetchChannel(input,rId, x, y) * scaleRGB + offsetRGB;
		float g = fetchChannel(input,gId, x, y) * scaleRGB + offsetRGB;
		float b = fetchChannel(input,bId, x, y) * scaleRGB + offsetRGB;
		float a = fetchChannel(input,aId, x, y) * scaleA + offsetA;

		output[y * input.size(3) + x] = rgbaToInt(r, g, b, a);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::selectOutputChannel(
	const torch::Tensor& inputTensor, GLubyte* outputBuffer,
	int r, int g, int b, int a, 
	float scaleRGB, float offsetRGB, float scaleA, float offsetA)
{
	if (inputTensor.size(0) != 1) throw std::exception("batch size must be one");
	if (inputTensor.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");
	unsigned width = inputTensor.size(3);
	unsigned height = inputTensor.size(2);
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, SelectOutputChannelKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	SelectOutputChannelKernel
		<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, 
		 inputTensor.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
		 reinterpret_cast<unsigned int*>(outputBuffer), r, g, b, a,
		 scaleRGB, offsetRGB, scaleA, offsetA);
	CUMAT_CHECK_ERROR();
	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void ScreenSpaceShadingKernel(
	dim3 virtual_size,
	torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> input,
	unsigned int* output,
	renderer::ShadingSettings settings)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)

		// read input
		float mask = input[0][0][y][x];
		float3 normal = make_float3(
			input[0][1][y][x],
			input[0][2][y][x],
			input[0][3][y][x]
		);
		normal = safeNormalize(normal);
		float ao = input[0][5][y][x];

		float3 color = make_float3(0);
		// ambient
		color += settings.ambientLightColor * settings.materialColor;
		// diffuse
		color += settings.diffuseLightColor * settings.materialColor * abs(dot(normal, settings.lightDirection));
		// specular
		//TODO
		// ambient occlusion
		color *= lerp(1, ao, settings.aoStrength);

		output[y * input.size(3) + x] = rgbaToInt(color.x, color.y, color.z, mask);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::screenShading(const torch::Tensor& inputTensor, GLubyte* outputBuffer, const renderer::ShadingSettings& settings)
{
	if (inputTensor.size(0) != 1) throw std::exception("batch size must be one");
	if (inputTensor.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");
	unsigned width = inputTensor.size(3);
	unsigned height = inputTensor.size(2);
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, ScreenSpaceShadingKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	ScreenSpaceShadingKernel
		<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size,
		 inputTensor.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
		 reinterpret_cast<unsigned int*>(outputBuffer), settings);
	CUMAT_CHECK_ERROR();
	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

#define MAX_LAYERS 16
__constant__ kernel::LayerData LayerDataGpu[MAX_LAYERS];

__global__ void FoveatedBlendingKernel(
	dim3 virtual_size,
	unsigned int* inoutBuffer,
	int numLayers)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)

		//background fullscreen color
		float4 color = intToFloat4(inoutBuffer[x + y * virtual_size.x ]);

		//blend each layer
		for (int i=0; i<numLayers; ++i)
		{
			if (x>=LayerDataGpu[i].viewport.x &&
				y>=LayerDataGpu[i].viewport.y &&
				x<LayerDataGpu[i].viewport.x+LayerDataGpu[i].viewport.z &&
				y<LayerDataGpu[i].viewport.y+LayerDataGpu[i].viewport.w)
			{
				//crab color
				float4 c;
				if (LayerDataGpu[i].subimage==nullptr)
				{
					//test case, use predefined colors
					static const float4 TEST_COLORS[] = { {1,0,0,1}, {0,1,0,1}, {0,0,1,1} };
					c = TEST_COLORS[i % 3];
				}
				else
				{
					c = intToFloat4(reinterpret_cast<unsigned int*>(LayerDataGpu[i].subimage)[
						(x - LayerDataGpu[i].viewport.x) +
						(y - LayerDataGpu[i].viewport.y) * LayerDataGpu[i].viewport.z]);
					//c.w = 1.0f;
				}
				//compute blending factor
				float2 pos = make_float2(
					(x - LayerDataGpu[i].viewport.x - 0.5f*LayerDataGpu[i].viewport.z) / (0.5f*LayerDataGpu[i].viewport.z),
					(y - LayerDataGpu[i].viewport.y - 0.5f*LayerDataGpu[i].viewport.w) / (0.5f*LayerDataGpu[i].viewport.w)
				);
				float r = length(pos);
				float s = LayerDataGpu[i].smoothingRadius;
				//float factor = clamp(1 - (r+s)*s, 0.0f, 1.0f);
				float factor = clamp(lerp(1.0f, 0.0f, (r - (1 - s)) / s), 0.0f, 1.0f);
				//apply shape function
				if (LayerDataGpu[i].blurShape == kernel::FoveatedBlurShapeSmoothstep)
					factor = (factor*factor*(3.0f - (2.0f*factor)));
				//blend it
				color = lerp(color, c, factor);
			}
		}

		//write result
		inoutBuffer[x + y * virtual_size.x] = float4ToInt(color);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::foveatedBlending(int width, int height, GLubyte* inoutBuffer, const std::vector<LayerData>& layers)
{
	if (layers.size() > MAX_LAYERS) throw std::exception("only " CUMAT_STR(MAX_LAYERS) " layers supported");
	if (layers.empty()) throw std::exception("at least one layer has to be specified");

	cuMat::Context& ctx = cuMat::Context::current();

	//copy layer info to the GPU
	cudaMemcpyToSymbolAsync(LayerDataGpu, 
	                        layers.data(), layers.size() * sizeof(LayerData), 0,
	                        cudaMemcpyHostToDevice, ctx.stream());

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, FoveatedBlendingKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	FoveatedBlendingKernel
		<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size,
		 reinterpret_cast<unsigned int*>(inoutBuffer), int(layers.size()));
	CUMAT_CHECK_ERROR();
	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void NetworkPostprocessingUnshadedKernel(
	dim3 virtual_size,
	torch::PackedTensorAccessor<float, 4> inout)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)

		// read input
		float mask = inout[0][0][y][x];
		float3 normal = make_float3(
			inout[0][1][y][x],
			inout[0][2][y][x],
			inout[0][3][y][x]
		);
		float depth = inout[0][4][y][x];
		float ao = inout[0][5][y][x];

		// process it
		mask = clamp(mask * 0.5f + 0.5f, 0.0f, 1.0f);
		normal = safeNormalize(normal);
		depth = clamp(depth, 0.f, 1.f);
		ao = clamp(ao, 0.f, 1.f);

		//if mask is low enough, we are outside and we set all other values to zero
		float inside = mask > 0.001 ? 1 : 0;
	
		// write output
		inout[0][0][y][x] = mask;
		inout[0][1][y][x] = normal.x * inside;
		inout[0][2][y][x] = normal.y * inside;
		inout[0][3][y][x] = normal.z * inside;
		inout[0][4][y][x] = depth * inside;
		inout[0][5][y][x] = ao * inside;

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::networkPostProcessingUnshaded(torch::Tensor& inout)
{
	if (inout.size(0) != 1) throw std::exception("batch size must be one");
	if (inout.size(1) != 6) throw std::exception("channel size must be six");
	if (inout.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");
	int width = inout.size(3);
	int height = inout.size(2);
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, NetworkPostprocessingUnshadedKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	NetworkPostprocessingUnshadedKernel
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, inout.packed_accessor<float, 4>());
	CUMAT_CHECK_ERROR();
}


__global__ void NetworkPostprocessingAdaptiveKernel(
	dim3 virtual_size,
	torch::PackedTensorAccessor<float, 4> inout)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)

		// read input
		float mask = inout[0][0][y][x];
		float3 normal = make_float3(
			inout[0][1][y][x],
			inout[0][2][y][x],
			inout[0][3][y][x]
		);
		float depth = inout[0][4][y][x];
		float ao = inout[0][5][y][x];
		float flowX = inout[0][6][y][x];
		float flowY = inout[0][7][y][x];

		// process it
		mask = clamp(mask * 0.5f + 0.5f, 0.0f, 1.0f);
		normal = safeNormalize(normal);
		depth = clamp(depth, 0.f, 1.f);
		ao = clamp(ao, 0.f, 1.f);
		flowX = clamp(flowX, -1.f, 1.f);
		flowY = clamp(flowY, -1.f, 1.f);

		// write output
		inout[0][0][y][x] = mask;
		inout[0][1][y][x] = normal.x;
		inout[0][2][y][x] = normal.y;
		inout[0][3][y][x] = normal.z;
		inout[0][4][y][x] = depth;
		inout[0][5][y][x] = ao;
		inout[0][6][y][x] = flowX;
		inout[0][7][y][x] = flowY;

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::networkPostProcessingAdaptive(torch::Tensor& inout)
{
	if (inout.size(0) != 1) throw std::exception("batch size must be one");
	if (inout.size(1) != 8) throw std::exception("channel size must be six");
	if (inout.dim() != 4) throw std::exception("tensor dimension must be 4 (B*C*H*W)");
	int width = inout.size(3);
	int height = inout.size(2);
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, NetworkPostprocessingAdaptiveKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	NetworkPostprocessingAdaptiveKernel
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, inout.packed_accessor<float, 4>());
	CUMAT_CHECK_ERROR();
}


torch::Tensor kernel::interpolateNearest(const torch::Tensor& inputTensor, int width, int height)
{
	return torch::native::upsample_nearest2d_cuda(
		inputTensor, { height, width });
}

torch::Tensor kernel::interpolateLinear(const torch::Tensor& inputTensor, int width, int height)
{
	return torch::native::upsample_bilinear2d_cuda(
		inputTensor, { height, width }, false);
}

torch::Tensor kernel::interpolateCubic(const torch::Tensor& inputTensor, int width, int height)
{
	return torch::native::upsample_bicubic2d_cuda(
		inputTensor, { height, width }, false);
}

torch::Tensor kernel::createInitialImage(const torch::Tensor& currentInput, int channels, InitialImage mode, int width,
                                         int height)
{
	switch (mode)
	{
	case InitialImageInput:
		{
			auto inputHeight = interpolateLinear(currentInput, width, height);
			if (inputHeight.size(1) == channels)
				return inputHeight;
			else if (inputHeight.size(1) > channels)
				return torch::cat({
					                  inputHeight,
					                  torch::ones({1, channels - inputHeight.size(1), height, width},
					                              torch::dtype(torch::kFloat).device(torch::kCUDA))
				                  }, 1);
			else
				throw std::exception("not supported yet");
		}
	case InitialImageUnshaded:
		{
			if (channels != 6) throw std::exception("channels must be 6 for InitialImageUnshaded");
			auto defaults = torch::tensor({ -1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 1.0f }, torch::dtype(torch::kFloat).device(torch::kCUDA));
			return defaults.view({ 1, channels, 1, 1 }).expand({ 1, channels, height, width });
		}
	case InitialImageZero:
		{
			return torch::zeros({ 1, channels, height, width }, torch::dtype(torch::kFloat).device(torch::kCUDA));
		}
	default: throw std::exception("unknown enum");
	}
}

torch::Tensor kernel::inpaintFlow(const torch::Tensor& mask, const torch::Tensor& data)
{
#if 0
	//move input to CPU
	torch::Tensor mask_cpu = (mask.detach() <= 0).to(torch::kCPU, torch::kUInt8);
	torch::Tensor data_cpu = data.detach().to(torch::kCPU);

	//create output tensor
	torch::Tensor output_cpu = torch::empty_like(data_cpu);
	
	//convert to OpenCV matrix
	int channel = data.size(1);
	int height = data.size(2);
	int width = data.size(3);
	cv::Mat mask_cv(height, width, CV_8U, mask_cpu.data_ptr());
	//perform inpainting
	for (int i = 0; i < channel; ++i)
	{
		cv::Mat data_cv(height, width, CV_32F,
			data_cpu.data<float>() + i*width*height);
		cv::Mat output_cv(height, width, CV_32F,
			output_cpu.data<float>() + i * width*height);
		cv::inpaint(data_cv, mask_cv, output_cv, 3, cv::INPAINT_NS);
	}

	//move output to GPU
	return output_cpu.to(data.device());
#else
	//fast inpaint on the GPU
	torch::Tensor mask2 = (mask - 0.5).squeeze(1);
	return renderer::Inpainting::fastInpaint(mask2, data);
#endif
}

torch::Tensor kernel::warpUpscale(const torch::Tensor& input_high, const torch::Tensor& flow_low, int upscale_factor)
{
	int C = flow_low.size(1);
	int H = flow_low.size(2);
	int W = flow_low.size(3);
	assert(C == 2);

	auto flow_xy = flow_low.chunk(2, 1);
	auto flow_low2 = torch::cat({ flow_xy[0] * -2.0, flow_xy[1] * -2.0 }, 1);

	auto flow_high = interpolateLinear(flow_low2, W*upscale_factor, H*upscale_factor);
	flow_high = flow_high.permute({ 0,2,3,1 });
	int Hhigh = H * upscale_factor;
	int Whigh = W * upscale_factor;

	typedef std::pair<int, int> tii;
	static std::map<tii, torch::Tensor> Offsets;
	const tii key(Hhigh, Whigh);
	torch::Tensor grid_offsets;
	if (Offsets.count(key)==0)
	{
		grid_offsets = torch::stack(torch::broadcast_tensors({
			                            torch::linspace(-1, +1, Whigh, at::device(flow_low.device()).dtype(flow_low.dtype()))
			                            .unsqueeze(0),
			                            torch::linspace(-1, +1, Hhigh, at::device(flow_low.device()).dtype(flow_low.dtype()))
			                            .unsqueeze(1)
		                            }), 2).unsqueeze_(0).detach_();
	}
	else
	{
		grid_offsets = Offsets.at(key);
	}
	auto grid = grid_offsets + flow_high;

	//TODO: special mask
	auto warped_high = torch::grid_sampler_2d(input_high, grid, 0 /*bilinear*/, 0 /*zeros*/, false);

	return warped_high;
}

void kernel::initializePyTorch()
{
	torch::Tensor t1 = torch::zeros({ 1,2,5,5 }, at::dtype(at::kFloat).device(at::kCUDA));
	torch::Tensor t2 = interpolateLinear(t1, 10, 10);
	(void)t2; 
}

__global__ void FillColorMapKernel(
	dim3 virtualSize,
	cudaSurfaceObject_t surface,
	cudaTextureObject_t tfTexture)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtualSize)

	auto density = x / static_cast<float>(virtualSize.x);
	auto rgbo = tex1D<float4>(tfTexture, density);

	surf2Dwrite(rgbaToInt(rgbo.x, rgbo.y, rgbo.z, 1.0f), surface, x * 4, y);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::fillColorMap(cudaSurfaceObject_t colorMap, cudaTextureObject_t tfTexture, int width, int height)
{
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, FillColorMapKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	FillColorMapKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, colorMap, tfTexture);
	CUMAT_CHECK_ERROR();
}

__global__ void TransferDvrOutputKernel(
	dim3 virtualSize,
	torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> dvrOutput,
	GLubyte* byteOutput)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtualSize)

	auto uintOutput = reinterpret_cast<unsigned int*>(byteOutput);
	uintOutput[y * virtualSize.x + x] = rgbaToInt(dvrOutput[0][0][y][x], dvrOutput[0][1][y][x], dvrOutput[0][2][y][x], 1.0f);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::transferDvrOutput(const at::Tensor& dvrOutput, GLubyte* byteOutput, int width, int height)
{
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, TransferDvrOutputKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	TransferDvrOutputKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, dvrOutput.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(), byteOutput);
	CUMAT_CHECK_ERROR();
}
