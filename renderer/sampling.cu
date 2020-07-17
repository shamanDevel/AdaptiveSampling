#include "sampling.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <glm/glm.hpp>
#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include "helper_math.h"

#ifdef RENDERER_HAS_SAMPLING
BEGIN_RENDERER_NAMESPACE

std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>
loadSamplesFromFile(const std::string& filename)
{
	int numVertices, numTriangles;
	std::ifstream f(filename);
	int64_t width, height;
	f >> width >> height;
	f >> numVertices >> numTriangles;

	torch::Tensor output = torch::empty({ 2, numVertices }, at::dtype(at::kFloat).device(at::kCPU));
	auto acc = output.accessor<float, 2>();
	for (int i=0; i<numVertices; ++i)
	{
		float x, y;
		f >> x >> y;
		if (!f) throw std::runtime_error("failed to read vertex");
		acc[0][i] = x;
		acc[1][i] = y;
	}
	
	torch::Tensor tris = torch::empty({ numTriangles, 3 }, at::dtype(at::kInt).device(at::kCPU));
	auto acc2 = tris.accessor<int, 2>();
	for (int i=0; i<numTriangles; ++i)
	{
		int3 tri;
		f >> tri.x >> tri.y >> tri.z;
		acc2[i][0] = tri.x;
		acc2[i][1] = tri.y;
		acc2[i][2] = tri.z;
		if (!f) throw std::runtime_error("failed to read triangle");
	}
	
	return std::make_tuple(output, tris, width, height);
}

#define CHECK_CPU(x) TORCH_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x.dtype() == at::kFloat), #x " must be a float tensor")
#define CHECK_INT(x) TORCH_CHECK((x.dtype() == at::kInt), #x " must be a int tensor")
#define CHECK_DIM(x, d) TORCH_CHECK((x.dim() == (d)), #x " must be a tensor with ", d, " dimensions, but has shape ", x.sizes())
#define CHECK_SIZE(x, d, s) TORCH_CHECK((x.size(d) == (s)), #x " must have ", s, " entries at dimension ", d, ", but has ", x.size(d), " entries")

torch::Tensor computeAverageSampleDistance(
	const torch::Tensor& samplePositions, 
	const torch::Tensor& sampleIndices)
{
	//check input
	CHECK_CPU(samplePositions);
	CHECK_CONTIGUOUS(samplePositions);
	CHECK_FLOAT(samplePositions);
	CHECK_DIM(samplePositions, 2);
	CHECK_SIZE(samplePositions, 0, 2);
	int64_t N = samplePositions.size(1);

	CHECK_CPU(sampleIndices);
	CHECK_CONTIGUOUS(sampleIndices);
	CHECK_INT(sampleIndices);
	CHECK_DIM(sampleIndices, 2);
	CHECK_SIZE(sampleIndices, 1, 3);
	int M = static_cast<int>(sampleIndices.size(0));

	//allocate output and temporary storage
	torch::Tensor output = torch::zeros({ N }, at::dtype(at::kFloat).device(at::kCPU));
	torch::Tensor count = torch::zeros({ N }, at::dtype(at::kFloat).device(at::kCPU));

	//create accessors
	const auto aPos = samplePositions.accessor<float, 2>();
	const auto aIndex = sampleIndices.accessor<int, 2>();
	auto aOut = output.accessor<float, 1>();
	auto aCount = count.accessor<float, 1>();
	
	//loop over triangles and compute the distances
	for (int m=0; m<M; ++m)
	{
		int i1 = aIndex[m][0];
		int i2 = aIndex[m][1];
		int i3 = aIndex[m][2];
		glm::vec2 p1(aPos[0][i1], aPos[1][i1]);
		glm::vec2 p2(aPos[0][i2], aPos[1][i2]);
		glm::vec2 p3(aPos[0][i3], aPos[1][i3]);
		float d12 = distance(p1, p2);
		float d13 = distance(p1, p3);
		float d23 = distance(p2, p3);
		aOut[i1] += d12 + d13;
		aOut[i2] += d12 + d23;
		aOut[i3] += d13 + d23;
		aCount[i1] += 2;
		aCount[i2] += 2;
		aCount[i3] += 2;
	}

	//std::cout << "Min count: " << torch::min(count) << ", max: " << torch::max(count) << std::endl;
	
	//normalize / average
	return output / count;
}

#define MAX_CHANNELS 16

template<typename scalar_t>
__global__ void AdaptiveSmoothingKernel(dim3 virtual_size,
	const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
	const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> distances,
	scalar_t distanceToStandardDeviation,
	torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output)
{
	const int height = input.size(2);
	const int width = input.size(3);
	const int channels = input.size(1);
	CUMAT_KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		const float2 p = make_float2(x, y);
		//get distance
		const scalar_t distance = distances[b][0][y][x];
		const scalar_t distSqr = distance * distance;
		const scalar_t standardDeviation = distance * distanceToStandardDeviation;
		const scalar_t denom = 1 / (2 * standardDeviation * standardDeviation);

		//accumulation buffer
		scalar_t out[MAX_CHANNELS] = { 0 };
		scalar_t weightSum = 0;

		//loop over influence radius
		for (int y_ = max(0, int(floor(y - distance))); y_ <= min(height - 1, int(ceil(y + distance))); ++y_)
		for (int x_ = max(0, int(floor(x - distance))); x_ <= min(width  - 1, int(ceil(x + distance))); ++x_)
		{
			//compute weight
			scalar_t d = lengthSquared(p - make_float2(x_, y_));
			if (d > distSqr) continue; //outside of the circle of radius 'distance'
			scalar_t weight = exp(-denom * d);
			//loop over channels
			for (int c=0; c<channels; ++c)
			{
				out[c] += weight * input[b][c][y_][x_];
			}
			weightSum += weight;
		}
		
		//write output
		for (int c=0; c<channels; ++c)
		{
			output[b][c][y][x] = out[c] / weightSum;
		}
	}
	CUMAT_KERNEL_3D_LOOP_END
}

torch::Tensor adaptiveSmoothing(
	const torch::Tensor& input, const torch::Tensor& distances,
	double distanceToStandardDeviation)
{
	//check input
	CHECK_CUDA(input);
	//CHECK_FLOAT(input);
	CHECK_CONTIGUOUS(input);
	CHECK_DIM(input, 4);
	int64_t B = input.size(0);
	int64_t C = input.size(1);
	int64_t H = input.size(2);
	int64_t W = input.size(3);
	TORCH_CHECK(C < 16, "AdaptiveSmoothing only supports up to 16 channels");

	CHECK_CUDA(distances);
	//CHECK_FLOAT(distances);
	CHECK_CONTIGUOUS(distances);
	CHECK_DIM(distances, 4);
	CHECK_SIZE(distances, 0, B);
	CHECK_SIZE(distances, 1, 1);
	CHECK_SIZE(distances, 2, H);
	CHECK_SIZE(distances, 3, W);
	
	TORCH_CHECK(distanceToStandardDeviation > 0, 
		"distanceToSTandardDeviation must be positive, but is ", distanceToStandardDeviation);

	//create output
	torch::Tensor output = torch::empty_like(input);

	//launch kernel
	cuMat::Context& ctx = cuMat::Context::current();
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	AT_DISPATCH_FLOATING_TYPES(input.type(), "AdaptiveSmoothingKernel", ([&]
	{
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
			W, H, B, AdaptiveSmoothingKernel<scalar_t>);
		AdaptiveSmoothingKernel<scalar_t>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				distances.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				distanceToStandardDeviation,
				output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
	}));
	CUMAT_CHECK_ERROR();
	return output;
}

template<typename scalar_t, bool WriteGradInput, bool WriteGradDistance>
__global__ void AdjointAdaptiveSmoothingKernel(dim3 virtual_size,
	const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
	const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> distances,
	scalar_t distanceToStandardDeviation,
	const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradOutput,
	torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> out_gradInput,
	torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> out_gradDistances)
{
	const int height = input.size(2);
	const int width = input.size(3);
	const int channels = input.size(1);
	CUMAT_KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		//------------------- FORWARD ----------------------
		const float2 p = make_float2(x, y);
		//get distance
		const scalar_t distance = distances[b][0][y][x];
		const scalar_t distSqr = distance * distance;
		const scalar_t standardDeviation = distance * distanceToStandardDeviation;
		const scalar_t denom = 1 / (2 * standardDeviation * standardDeviation);

		//accumulation buffer
		scalar_t out[MAX_CHANNELS] = { 0 };
		scalar_t weightSum = 0;

		//loop over influence radius
		for (int y_ = max(0, int(floor(y - distance))); y_ <= min(height - 1, int(ceil(y + distance))); ++y_)
		for (int x_ = max(0, int(floor(x - distance))); x_ <= min(width - 1, int(ceil(x + distance))); ++x_)
		{
			//compute weight
			scalar_t d = lengthSquared(p - make_float2(x_, y_));
			if (d > distSqr) continue; //outside of the circle of radius 'distance'
			scalar_t weight = exp(-denom * d);
			//loop over channels
			for (int c = 0; c < channels; ++c)
			{
				out[c] += weight * input[b][c][y_][x_];
			}
			weightSum += weight;
		}

		////write output
		//for (int c = 0; c < channels; ++c)
		//{
		//	output[b][c][y][x] = out[c] / weightSum;
		//}

		//------------------- BACKWARD ----------------------
#define IDX(tensor, b,c,y,x) ((b)*tensor.stride(0)+(c)*tensor.stride(1)+(y)*tensor.stride(2)+(x)*tensor.stride(3))
		
		scalar_t adjWeightSum = 0;
		scalar_t adjOut[MAX_CHANNELS] = { 0 };
		for (int c=0; c<channels; ++c)
		{
			adjOut[c] = gradOutput[b][c][y][x] / weightSum;
			adjWeightSum -= gradOutput[b][c][y][x] * out[c] / (weightSum*weightSum);
		}

		scalar_t adjDenom = 0;
		for (int y_ = max(0, int(floor(y - distance))); y_ <= min(height - 1, int(ceil(y + distance))); ++y_)
		for (int x_ = max(0, int(floor(x - distance))); x_ <= min(width - 1, int(ceil(x + distance))); ++x_)
		{
			//compute weight
			scalar_t d = lengthSquared(p - make_float2(x_, y_));
			if (d > distSqr) continue; //outside of the circle of radius 'distance'
			scalar_t weight = exp(-denom * d);
			//adjoint code
			scalar_t adjWeight = adjWeightSum;
			for (int c=0; c<channels; ++c)
			{
				adjWeight += adjOut[c] * input[b][c][y_][x_];
				if (WriteGradInput) {
					atomicAdd(out_gradInput.data() + IDX(out_gradInput, b, c, y_, x_), adjOut[c] * weight);
				}
			}
			adjDenom -= adjWeight * d * exp(-denom * d);
		}

		if (WriteGradDistance) {
			scalar_t adjStandardDeviation = -adjDenom / (standardDeviation*standardDeviation*standardDeviation);
			scalar_t adjDistance = adjStandardDeviation *distanceToStandardDeviation;
			atomicAdd(out_gradDistances.data() + IDX(out_gradDistances, b, 0, y, x), adjDistance);
		}
#undef IDX
	}
	CUMAT_KERNEL_3D_LOOP_END
}

std::tuple<torch::Tensor, torch::Tensor> adjAdaptiveSmoothing(const torch::Tensor& input, const torch::Tensor& distances,
	double distanceToStandardDeviation, const torch::Tensor& gradOutput, 
	bool writeGradInput, bool writeGradDistances)
{
	//check input
	CHECK_CUDA(input);
	CHECK_CONTIGUOUS(input);
	CHECK_DIM(input, 4);
	int64_t B = input.size(0);
	int64_t C = input.size(1);
	int64_t H = input.size(2);
	int64_t W = input.size(3);
	TORCH_CHECK(C < 16, "AdaptiveSmoothing only supports up to 16 channels");

	CHECK_CUDA(distances);
	CHECK_CONTIGUOUS(distances);
	CHECK_DIM(distances, 4);
	CHECK_SIZE(distances, 0, B);
	CHECK_SIZE(distances, 1, 1);
	CHECK_SIZE(distances, 2, H);
	CHECK_SIZE(distances, 3, W);

	TORCH_CHECK(distanceToStandardDeviation > 0,
		"distanceToSTandardDeviation must be positive, but is ", distanceToStandardDeviation);

	CHECK_CUDA(gradOutput);
	CHECK_CONTIGUOUS(gradOutput);
	CHECK_DIM(gradOutput, 4);
	CHECK_SIZE(gradOutput, 0, B);
	CHECK_SIZE(gradOutput, 1, C);
	CHECK_SIZE(gradOutput, 2, H);
	CHECK_SIZE(gradOutput, 3, W);
	
	//create output
	torch::Tensor out_gradInput = torch::zeros_like(input);
	torch::Tensor out_gradDistances = torch::zeros_like(distances);

	//create output
	torch::Tensor output = torch::empty_like(input);

	//launch kernel
	cuMat::Context& ctx = cuMat::Context::current();
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH(gradInput, gradDistances) {															\
	AT_DISPATCH_FLOATING_TYPES(input.type(), "AdjointAdaptiveSmoothingKernel", ([&]						\
	{																									\
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(										\
			W, H, B, AdjointAdaptiveSmoothingKernel<scalar_t, gradInput, gradDistances>);				\
		AdjointAdaptiveSmoothingKernel<scalar_t, gradInput, gradDistances>								\
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>									\
			(cfg.virtual_size,																			\
				input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),					\
				distances.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),				\
				distanceToStandardDeviation,															\
				gradOutput.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),			\
				out_gradInput.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),			\
				out_gradDistances.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());	\
	}));																								\
	}
	switch (writeGradInput)
	{
	case false: {
		switch (writeGradDistances)
		{
		case false: DISPATCH(false, false); break;
		case true: DISPATCH(false, true); break;
		}} break;
	case true: {
		switch (writeGradDistances)
		{
		case false: DISPATCH(true, false); break;
		case true: DISPATCH(true, true); break;
		}} break;
	}
	
	CUMAT_CHECK_ERROR();
	return std::make_tuple(out_gradInput, out_gradDistances);
}

END_RENDERER_NAMESPACE
#endif
