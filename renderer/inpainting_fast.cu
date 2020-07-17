#include "inpainting.h"

#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stack>

#ifdef RENDERER_HAS_INPAINTING

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, d) TORCH_CHECK((x.dim() == (d)), #x " must be a tensor with ", d, " dimensions, but has shape ", x.sizes())
#define CHECK_SIZE(x, d, s) TORCH_CHECK((x.size(d) == (s)), #x " must have ", s, " entries at dimension ", d, ", but has ", x.size(d), " entries")

#define MAX_CHANNELS 16

///////////////////////////////////////////////////////////////////////
// fast inpainting - discrete version
///////////////////////////////////////////////////////////////////////

namespace
{
	__device__ inline int start_index(int a, int b, int c) {
		return (int)floor((float)(a * c) / b);
	}

	__device__ inline int end_index(int a, int b, int c) {
		return (int)ceil((float)((a + 1) * c) / b);
	}

	template<typename scalar_t>
	__global__ void FastInpaintingKernel_Down(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mask,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> data,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLow,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLow)
	{
		const int H = mask.size(1);
		const int W = mask.size(2);
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
		{
			int N = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
			for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
				for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
				{
					if (mask[b][ii][jj] >= 0.5)
					{
						N++;
						for (int c = 0; c < C; ++c)
							d[c] += data[b][c][ii][jj];
					}
				}
			maskLow[b][i][j] = N > 0 ? 1 : 0;
			for (int c = 0; c < C; ++c)
				dataLow[b][c][i][j] = N > 0 ? d[c] / N : 0;
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	template<typename scalar_t>
	__global__ void FastInpaintingKernel_Up(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mask,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> data,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLow,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLow,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskHigh,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataHigh)
	{
		const int H = mask.size(1);
		const int W = mask.size(2);
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
		{
			if (mask[b][i][j] >= 0.5)
			{
				//copy unchanged
				maskHigh[b][i][j] = 1;
				for (int c = 0; c < C; ++c)
					dataHigh[b][c][i][j] = data[b][c][i][j];
			}
			else
			{
				//interpolate from low resolution (bilinear)
				//get neighbor offsets
				int io = i % 2 == 0 ? -1 : +1;
				int jo = j % 2 == 0 ? -1 : +1;
				//accumulate
				scalar_t N = 0;
				scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)													\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW && maskLow[b][(ii)][(jj)]>=0.5) {	\
		N += w;															\
		for (int c = 0; c < C; ++c) d[c] += w * dataLow[b][c][(ii)][(jj)];		\
	}
				ITEM(i / 2, j / 2, 0.75f*0.75f);
				ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
				ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
				ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
#undef ITEM
				//write output
				maskHigh[b][i][j] = N > 0 ? 1 : 0;
				for (int c = 0; c < C; ++c)
					dataHigh[b][c][i][j] = N > 0 ? d[c] / N : 0;
			}
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	std::tuple<torch::Tensor, torch::Tensor>
		fastInpaint_recursion(
			const torch::Tensor& mask,
			const torch::Tensor& data)
	{
		int64_t B = data.size(0);
		int64_t C = data.size(1);
		int64_t H = data.size(2);
		int64_t W = data.size(3);

		if (H <= 1 && W <= 1)
			return std::make_tuple(mask, data); //end of recursion

		int64_t oH = H / 2;
		int64_t oW = W / 2;

		//prepare launching
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		//downsample
		torch::Tensor maskLow = torch::empty({ B, oH, oW }, mask.options());
		torch::Tensor dataLow = torch::empty({ B, C, oH, oW }, data.options());
		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Down", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				oW, oH, B, FastInpaintingKernel_Down<scalar_t>);
			FastInpaintingKernel_Down<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLow.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLow.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//recursion
		const auto tuple = fastInpaint_recursion(maskLow, dataLow);
		const auto& maskLow2 = std::get<0>(tuple);
		const auto& dataLow2 = std::get<1>(tuple);

		//upsample
		torch::Tensor maskHigh = torch::empty({ B, H, W }, mask.options());
		torch::Tensor dataHigh = torch::empty({ B, C, H, W }, data.options());
		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Up", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				W, H, B, FastInpaintingKernel_Up<scalar_t>);
			FastInpaintingKernel_Up<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLow2.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLow2.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskHigh.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataHigh.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//done
		return std::make_tuple(maskHigh, dataHigh);
	}

}

torch::Tensor renderer::Inpainting::fastInpaint(
	const torch::Tensor& mask,
	const torch::Tensor& data)
{
	//check input
	CHECK_CUDA(mask);
	CHECK_CONTIGUOUS(mask);
	CHECK_DIM(mask, 3);
	int64_t B = mask.size(0);
	int64_t H = mask.size(1);
	int64_t W = mask.size(2);

	CHECK_CUDA(data);
	CHECK_CONTIGUOUS(data);
	CHECK_DIM(data, 4);
	CHECK_SIZE(data, 0, B);
	int64_t C = data.size(1);
	CHECK_SIZE(data, 2, H);
	CHECK_SIZE(data, 3, W);
	TORCH_CHECK(C < 16, "Inpainting::fastInpaint only supports up to 16 channels, but got " + std::to_string(C));

	//inpaint recursivly
	return std::get<1>(fastInpaint_recursion(mask, data));
}


///////////////////////////////////////////////////////////////////////
// fast inpainting - fractional version
///////////////////////////////////////////////////////////////////////

namespace
{

	template<typename scalar_t>
	__global__ void FastInpaintingFractionalKernel_Down(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mask,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> data,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLow,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLow)
	{
		const int H = mask.size(1);
		const int W = mask.size(2);
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
		{
			int Count = 0;
			scalar_t N1 = 0;
			scalar_t N2 = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
			for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
				for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
				{
					Count++;
					N1 += mask[b][ii][jj];
					N2 = max(N2, mask[b][ii][jj]);
					for (int c = 0; c < C; ++c)
						d[c] += mask[b][ii][jj] * data[b][c][ii][jj];
				}
			//maskLow[b][i][j] = N1 / Count;
			maskLow[b][i][j] = N2;
			for (int c = 0; c < C; ++c)
				dataLow[b][c][i][j] = N1 > 0 ? d[c] / N1 : 0;
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	template<typename scalar_t>
	__global__ void FastInpaintingFractionalKernel_Up(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mask,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> data,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLow,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLow,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskHigh,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataHigh)
	{
		const int H = mask.size(1);
		const int W = mask.size(2);
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of high resolution
		{
			//interpolate from low resolution (bilinear)
			//get neighbor offsets
			int io = i % 2 == 0 ? -1 : +1;
			int jo = j % 2 == 0 ? -1 : +1;
			//accumulates
			scalar_t Weight = 0;
			scalar_t N = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)														\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW) {								\
		Weight += w;															\
		N += w * maskLow[b][(ii)][(jj)];										\
		for (int c = 0; c < C; ++c)												\
			d[c] += w * maskLow[b][(ii)][(jj)] * dataLow[b][c][(ii)][(jj)];		\
	}
			ITEM(i / 2, j / 2, 0.75f*0.75f);
			ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
			ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
			ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
#undef ITEM
			//write output
			scalar_t m = mask[b][i][j];
			maskHigh[b][i][j] = m + (N > 0 ? (1 - m) * (N / Weight) : 0);
			for (int c = 0; c < C; ++c)
			{
				dataHigh[b][c][i][j] =
					m * data[b][c][i][j] +
					(1 - m) * (N > 0 ? d[c] / N : 0);
			}
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	std::tuple<torch::Tensor, torch::Tensor>
		fastInpaintFractional_recursion(
			const torch::Tensor& mask,
			const torch::Tensor& data,
			bool saveResults,
			std::stack<torch::Tensor>& iStack)
	{
		int64_t B = data.size(0);
		int64_t C = data.size(1);
		int64_t H = data.size(2);
		int64_t W = data.size(3);

		//std::cout << "fastInpaintFractional_recursion - Pre:"
		//	<< " shape=(" << H << ", " << W << ")"
		//	<< ", data min=" << torch::min(data).item().toFloat()
		//	<< ", max=" << torch::max(data).item().toFloat()
		//	<< ", avg=" << torch::mean(data).item().toFloat()
		//	<< std::endl;

		if (H <= 1 || W <= 1)
			return std::make_tuple(mask, data); //end of recursion

		int64_t oH = H / 2;
		int64_t oW = W / 2;

		//prepare launching
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		//downsample
		torch::Tensor maskLowPre = torch::empty({ B, oH, oW }, mask.options());
		torch::Tensor dataLowPre = torch::empty({ B, C, oH, oW }, data.options());
		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Down", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				oW, oH, B, FastInpaintingFractionalKernel_Down<scalar_t>);
			FastInpaintingFractionalKernel_Down<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLowPre.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLowPre.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//recursion
		const auto tuple = fastInpaintFractional_recursion(maskLowPre, dataLowPre, saveResults, iStack);
		const auto& maskLowPost = std::get<0>(tuple);
		const auto& dataLowPost = std::get<1>(tuple);

		//upsample
		torch::Tensor maskHigh = torch::empty({ B, H, W }, mask.options());
		torch::Tensor dataHigh = torch::empty({ B, C, H, W }, data.options());
		AT_DISPATCH_FLOATING_TYPES(data.type(), "FastInpaintingKernel_Up", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				W, H, B, FastInpaintingFractionalKernel_Up<scalar_t>);
			FastInpaintingFractionalKernel_Up<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					mask.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLowPost.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLowPost.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskHigh.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataHigh.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//save for adjoint
		if (saveResults)
		{
			iStack.push(maskLowPre);
			iStack.push(dataLowPre);
			iStack.push(maskLowPost);
			iStack.push(dataLowPost);
		}

		//done
		return std::make_tuple(maskHigh, dataHigh);
	}

}

torch::Tensor renderer::Inpainting::fastInpaintFractional(
	const torch::Tensor& mask,
	const torch::Tensor& data)
{
	//check input
	CHECK_CUDA(mask);
	CHECK_CONTIGUOUS(mask);
	CHECK_DIM(mask, 3);
	int64_t B = mask.size(0);
	int64_t H = mask.size(1);
	int64_t W = mask.size(2);

	CHECK_CUDA(data);
	CHECK_CONTIGUOUS(data);
	CHECK_DIM(data, 4);
	CHECK_SIZE(data, 0, B);
	int64_t C = data.size(1);
	CHECK_SIZE(data, 2, H);
	CHECK_SIZE(data, 3, W);
	TORCH_CHECK(C < 16, "Inpainting::fastInpaint only supports up to 16 channels, but got " + std::to_string(C));

	//inpaint recursivly
	std::stack<torch::Tensor> s;
	return std::get<1>(fastInpaintFractional_recursion(mask, data, false, s));
}

///////////////////////////////////////////////////////////////////////
// fast inpainting - adjoint
///////////////////////////////////////////////////////////////////////

namespace
{
#define IDX4(tensor, b,c,y,x) ((b)*tensor.stride(0)+(c)*tensor.stride(1)+(y)*tensor.stride(2)+(x)*tensor.stride(3))
#define IDX3(tensor, b,y,x) ((b)*tensor.stride(0)+(y)*tensor.stride(1)+(x)*tensor.stride(2))

	template<typename scalar_t>
	__global__ void AdjFastInpaintingFractionKernel_Down(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskIn,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataIn,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gradMaskLowIn,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradDataLowIn,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gradMaskOut,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradDataOut)
	{
		const int H = maskIn.size(1);
		const int W = maskIn.size(2);
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = dataIn.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of low resolution
		{
			//forward
			int Count = 0;
			scalar_t N1 = 0;
			scalar_t N2 = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
			for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
				for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
				{
					Count++;
					N1 += maskIn[b][ii][jj];
					N2 = max(N2, maskIn[b][ii][jj]);
					for (int c = 0; c < C; ++c)
						d[c] += maskIn[b][ii][jj] * dataIn[b][c][ii][jj];
				}
			//maskLow[b][i][j] = N2;
			////maskLow[b][i][j] = N1 / Count;
			//for (int c = 0; c < C; ++c)
			//	dataLow[b][c][i][j] = N1 > 0 ? d[c] / N1 : 0;

			//adjoint
			//Note: no atomics since every high-res pixel is accessed only once
			scalar_t adjD[MAX_CHANNELS] = { 0 };
			scalar_t adjN1 = 0;
			for (int c = 0; c < C; ++c)
			{
				adjD[c] = N1 > 0 ? gradDataLowIn[b][c][i][j] / N1 : 0;
				adjN1 -= N1 > 0 ? gradDataLowIn[b][c][i][j] * d[c] / (N1*N1) : 0;
			}
			scalar_t adjN2 = gradMaskLowIn[b][i][j];
			for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
				for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
				{
					scalar_t adjMask = 0;
					for (int c = 0; c < C; ++c)
					{
						gradDataOut[b][c][ii][jj] += adjD[c] * maskIn[b][ii][jj];
						adjMask += adjD[c] * dataIn[b][c][ii][jj];
					}
					adjMask += adjN1;
					//N2 = max(N2, maskIn[b][ii][jj]);
					if (N2 == maskIn[b][ii][jj])
						adjMask += adjN2;

					gradMaskOut[b][ii][jj] += adjMask;
				}
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	template<typename scalar_t>
	__global__ void AdjFastInpaintingFractionalKernel_Up(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskIn,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataIn,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> maskLowIn,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dataLowIn,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gradMaskHighIn,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradDataHighIn,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gradMaskOut,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradDataOut,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gradMaskLowOut,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> gradDataLowOut)
	{
		const int H = maskIn.size(1);
		const int W = maskIn.size(2);
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = dataIn.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: size of high resolution
		{
			//FORWARD

			//interpolate from low resolution (bilinear)
			//get neighbor offsets
			int io = i % 2 == 0 ? -1 : +1;
			int jo = j % 2 == 0 ? -1 : +1;
			//accumulates
			scalar_t Weight = 0;
			scalar_t N = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)															\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW) {								\
		Weight += w;															\
		N += w * maskLowIn[b][(ii)][(jj)];										\
		for (int c = 0; c < C; ++c)												\
			d[c] += w * maskLowIn[b][(ii)][(jj)] * dataLowIn[b][c][(ii)][(jj)];	\
	}
			ITEM(i / 2, j / 2, 0.75f*0.75f);
			ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
			ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
			ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
#undef ITEM
			//write output
			scalar_t m = maskIn[b][i][j];
			//maskHigh[b][i][j] = m + (1 - m) * (N / Weight);
			//for (int c = 0; c < C; ++c)
			//{
			//	dataHigh[b][c][i][j] =
			//		m * dataIn[b][c][i][j] +
			//		(1 - m) * (N > 0 ? d[c] / N : 0);
			//}

			//ADJOINT

			scalar_t adjD[MAX_CHANNELS] = { 0 };
			scalar_t adjMask = 0;
			scalar_t adjN = 0;
			for (int c = 0; c < C; ++c)
			{
				const scalar_t adjDataHigh = gradDataHighIn[b][c][i][j];
				//dataHigh[b][c][i][j] = m * dataIn[b][c][i][j] + (1 - m) * (N > 0 ? d[c] / N : 0);
				adjMask += adjDataHigh * (dataIn[b][c][i][j] - (N > 0 ? d[c] / N : 0));
				gradDataOut[b][c][i][j] += adjDataHigh * m;
				adjN -= N > 0 ? adjDataHigh * (1 - m)*d[c] / (N*N) : 0;
				adjD[c] += N > 0 ? adjDataHigh * (1 - m) / N : 0;
			}
			const scalar_t adjMaskHigh = gradMaskHighIn[b][i][j];
			//maskHigh[b][i][j] = m + (1 - m) * (N / Weight);
			adjMask += adjMaskHigh * (1 - (N / Weight));
			adjN += adjMaskHigh * ((1 - m) / Weight);
			gradMaskOut[b][i][j] += adjMask;

#define ITEM(ii,jj,w)																					\
			if ((ii) >= 0 && (jj) >= 0 && (ii) < oH && (jj) < oW) {										\
				scalar_t adjMaskLow = 0;																\
				for (int c=0; c<C; ++c)																	\
				{																						\
					/* d[c] += w * maskLowIn[b][(ii)][(jj)] * dataLowIn[b][c][(ii)][(jj)]; */			\
					adjMaskLow += adjD[c] * w * dataLowIn[b][c][(ii)][(jj)];							\
					atomicAdd(gradDataLowOut.data() + IDX4(gradDataLowOut, b, c, (ii), (jj)),			\
						adjD[c] * w * maskLowIn[b][(ii)][(jj)]);										\
				}																						\
				/* N += w * maskLowIn[b][(ii)][(jj)]; */												\
				adjMaskLow += adjN * w;																	\
				atomicAdd(gradMaskLowOut.data() + IDX3(gradMaskLowOut, b, (ii), (jj)), adjMaskLow);		\
			}
			ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
			ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
			ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
			ITEM(i / 2, j / 2, 0.75f*0.75f);
#undef ITEM
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

#undef IDX3
#undef IDX4

	void
		adjFastInpaintFractional_recursion(
			const torch::Tensor& maskIn,  // all in high resolution
			const torch::Tensor& dataIn,
			const torch::Tensor& gradMaskIn,
			const torch::Tensor& gradDataIn,
			torch::Tensor& gradMaskOut,
			torch::Tensor& gradDataOut,
			std::stack<torch::Tensor>& iStack)
	{
		int64_t B = dataIn.size(0);
		int64_t C = dataIn.size(1);
		int64_t H = dataIn.size(2);
		int64_t W = dataIn.size(3);
		int64_t oH = H / 2;
		int64_t oW = W / 2;

		if (H <= 1 || W <= 1)
		{
			//end of recursion
			gradMaskOut += gradMaskIn;
			gradDataOut += gradDataIn;
			return;
		}

		//get saved tensors (from after recursion in the forward pass
		torch::Tensor dataLowPost = iStack.top(); iStack.pop();
		torch::Tensor maskLowPost = iStack.top(); iStack.pop();
		torch::Tensor dataLowPre = iStack.top(); iStack.pop();
		torch::Tensor maskLowPre = iStack.top(); iStack.pop();

		//prepare launching
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		//adjoint upsample
		torch::Tensor gradMaskLowPost = torch::zeros({ B, oH, oW }, maskIn.options());
		torch::Tensor gradDataLowPost = torch::zeros({ B, C, oH, oW }, dataIn.options());
		AT_DISPATCH_FLOATING_TYPES(dataIn.type(), "AdjFastInpaintingFractionalKernel_Up", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				W, H, B, AdjFastInpaintingFractionalKernel_Up<scalar_t>);
			AdjFastInpaintingFractionalKernel_Up<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					maskIn.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataIn.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskLowPost.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataLowPost.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					gradMaskIn.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					gradDataIn.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					gradMaskOut.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					gradDataOut.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					gradMaskLowPost.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					gradDataLowPost.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//recursion
		torch::Tensor gradMaskLowPre = torch::zeros({ B, H, W }, maskIn.options());
		torch::Tensor gradDataLowPre = torch::zeros({ B, C, H, W }, dataIn.options());
		adjFastInpaintFractional_recursion(
			maskLowPre,
			dataLowPre,
			gradMaskLowPost,
			gradDataLowPost,
			gradMaskLowPre,
			gradDataLowPre,
			iStack);

		//adjoint downsample
		AT_DISPATCH_FLOATING_TYPES(dataIn.type(), "AdjFastInpaintingFractionKernel_Down", ([&]
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				oW, oH, B, AdjFastInpaintingFractionKernel_Down<scalar_t>);
			AdjFastInpaintingFractionKernel_Down<scalar_t>
				<< < cfg.block_count, cfg.thread_per_block, 0, stream >> >
				(cfg.virtual_size,
					maskIn.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					dataIn.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					gradMaskLowPre.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					gradDataLowPre.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					gradMaskOut.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
					gradDataOut.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		}));
		CUMAT_CHECK_ERROR();

		//done
	}

}



std::tuple<torch::Tensor, torch::Tensor> renderer::Inpainting::adjFastInpaintingFractional(
	const torch::Tensor& mask, const torch::Tensor& data,
	const torch::Tensor& gradOutput)
{
	//check input
	CHECK_CUDA(mask);
	CHECK_CONTIGUOUS(mask);
	CHECK_DIM(mask, 3);
	int64_t B = mask.size(0);
	int64_t H = mask.size(1);
	int64_t W = mask.size(2);

	CHECK_CUDA(data);
	CHECK_CONTIGUOUS(data);
	CHECK_DIM(data, 4);
	CHECK_SIZE(data, 0, B);
	int64_t C = data.size(1);
	CHECK_SIZE(data, 2, H);
	CHECK_SIZE(data, 3, W);
	TORCH_CHECK(C < 16, "Inpainting::fastInpaint only supports up to 16 channels, but got " + std::to_string(C));

	CHECK_CUDA(gradOutput);
	CHECK_CONTIGUOUS(gradOutput);
	CHECK_DIM(gradOutput, 4);
	CHECK_SIZE(gradOutput, 0, B);
	CHECK_SIZE(gradOutput, 1, C);
	CHECK_SIZE(gradOutput, 2, H);
	CHECK_SIZE(gradOutput, 3, W);

	//create output
	torch::Tensor outGradMask = torch::zeros_like(mask);
	torch::Tensor outGradData = torch::zeros_like(data);

	//run forward again, but save results
	std::stack<torch::Tensor> s;
	fastInpaintFractional_recursion(mask, data, true, s);

	//run adjoint code
	torch::Tensor gradMask = torch::zeros_like(mask);
	adjFastInpaintFractional_recursion(
		mask, data,
		gradMask, gradOutput,
		outGradMask, outGradData,
		s);

	return std::make_tuple(outGradMask, outGradData);
}

#endif