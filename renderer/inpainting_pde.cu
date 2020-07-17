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
// PDE inpainting - discrete version
///////////////////////////////////////////////////////////////////////

namespace
{
	int next_power_of_two(int v)
	{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}

	template<typename scalar_t>
	__global__ void MultigridSmooth(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> u,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> f,
		const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> mask,
		const scalar_t h,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> uOut)
	{
		const int H = u.size(2);
		const int W = u.size(3);
		const int C = u.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: W x H x B
		{
			if (mask[b][i][j])
			{
				//dirichlet boundary
				for (int c = 0; c < C; ++c)
					uOut[b][c][i][j] = f[b][c][i][j];
			} else
			{
				//free variable
				for (int c=0; c<C; ++c)
				{
					scalar_t v = 0;
					scalar_t uv = u[b][c][i][j];
					v += i > 0 ? u[b][c][i-1][j] : uv;
					v += j > 0 ? u[b][c][i][j-1] : uv;
					v += i < H-1 ? u[b][c][i+1][j] : uv;
					v += j < W-1 ? u[b][c][i][j+1] : uv;
					uOut[b][c][i][j] = (-h / 4) * (f[b][c][i][j] - (1 / h)*v);
				}
			}
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	//per-block full solving. Block Size: WxHx1
	//Requires WxHxC shared memory of type scalar_t
	template<typename scalar_t>
	__global__ void MultigridSmoothBlock(
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> u,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> f,
		const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> mask,
		const scalar_t h, const int iterations, const int b,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> uOut)
	{
		//const int b = blockIdx.z; //batch
		const int j = threadIdx.x; //position
		const int i = threadIdx.y;
		const int H = u.size(2);
		const int W = u.size(3);
		const int C = u.size(1);

		//copy u into shared memory
		extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
		scalar_t* uLocal = reinterpret_cast<scalar_t*>(my_smem);
#define IDX(c,ii,jj) ((jj) + W*((ii) + H*(c)))
		for (int c = 0; c < C; ++c)
			uLocal[IDX(c, i, j)] = u[b][c][i][j];
		__syncthreads();

		//copy f and mask into local memory
		scalar_t fLocal[MAX_CHANNELS];
		for (int c = 0; c < C; ++c)
			fLocal[c] = f[b][c][i][j];
		bool maskLocal = mask[b][i][j];
		
		//run Jacobi iterations
		for (int iter = 0; iter < iterations; ++iter)
		{
			for (int c = 0; c < C; ++c)
			{
				//compute new value
				scalar_t v = 0;
				if (maskLocal)
					v = fLocal[c];
				else
				{
					scalar_t uv = uLocal[IDX(c, i, j)];
					v += i > 0 ? uLocal[IDX(c, i-1, j)] : uv;
					v += j > 0 ? uLocal[IDX(c, i, j-1)] : uv;
					v += i < H - 1 ? uLocal[IDX(c, i+1, j)] : uv;
					v += j < W - 1 ? uLocal[IDX(c, i, j+1)] : uv;
					v = (-h / 4) * (fLocal[c] - (1 / (h))*v);
				}
				__syncthreads();

				//write new value
				uLocal[IDX(c, i, j)] = v;
				__syncthreads();
			}
		}

		//copy back
		for (int c = 0; c < C; ++c)
			uOut[b][c][i][j] = uLocal[IDX(c, i, j)];

#undef IDX
	}

	//Loop over W x H x B
	template<typename scalar_t>
	__global__ void MultigridResidual(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> u,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> f,
		const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> mask,
		const scalar_t h,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> residual)
	{
		const int H = u.size(2);
		const int W = u.size(3);
		const int C = u.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: W x H x B
		{
			if (mask[b][i][j])
			{
				//dirichlet boundary
				for (int c = 0; c < C; ++c)
					residual[b][c][i][j] = f[b][c][i][j] - u[b][c][i][j];
			}
			else
			{
				//free variable
				for (int c = 0; c < C; ++c)
				{
					scalar_t v = 0;
					scalar_t uv = u[b][c][i][j];
					v += i > 0 ? u[b][c][i - 1][j] : uv;
					v += j > 0 ? u[b][c][i][j - 1] : uv;
					v += i < H - 1 ? u[b][c][i + 1][j] : uv;
					v += j < W - 1 ? u[b][c][i][j + 1] : uv;
					residual[b][c][i][j] = f[b][c][i][j] - (1/(h))*(v-4*uv);
				}
			}
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	//Loop over Wlow x Hlow x B
	template<typename scalar_t>
	__global__ void MultigridRestrict(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> u,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> f,
		const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> mask,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> uLow,
		torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> maskLow)
	{
		const int H = u.size(2);
		const int W = u.size(3);
		const int C = u.size(1);
		CUMAT_KERNEL_3D_LOOP(jLow, iLow, b, virtual_size) //virtual_size: Wlow x Hlow x B
		{
			scalar_t vSmall[MAX_CHANNELS] = { 0 };
			scalar_t mSmall = 0;
			int counter = 0;

			//          [ 1 2 1 ]
			// 1 / 16 * [ 2 4 2 ]
			//          [ 1 2 1 ]
#define ENTRY(ii,jj, factor) { \
			for (int c=0; c<C; ++c) vSmall[c] += factor * u[b][c][ii][jj]; \
			mSmall += factor * (mask[b][ii][jj] ? 1.0f : 0.0f); \
			counter += factor; \
		}
			const int i = 2 * iLow;
			const int j = 2 * jLow;
			ENTRY(i, j, 4);
			if (i > 0) ENTRY(i - 1, j, 2);
			if (j > 0) ENTRY(i, j - 1, 2);
			if (i < H-1) ENTRY(i + 1, j, 2);
			if (j < W-1) ENTRY(i, j + 1, 2);
			if (i > 0 && j > 0) ENTRY(i - 1, j - 1, 1);
			if (i < H - 1 && j>0) ENTRY(i + 1, j - 1, 1);
			if (i > 0 && j < W - 1) ENTRY(i - 1, j + 1, 1);
			if (i < H - 1 && j < W - 1) ENTRY(i + 1, j + 1, 1);

			bool newMask = (mSmall / counter) > 0.1;
			maskLow[b][i][j] = newMask;
			for (int c = 0; c < C; ++c)
				uLow[b][c][i][j] = newMask ? scalar_t(0) : vSmall[c] / counter;
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	//Loop over Whigh x Hhigh x B
	template<typename scalar_t>
	__global__ void MultigridInterpolate(dim3 virtual_size,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> uLow,
		const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> maskHigh,
		torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> uHigh)
	{
		const int C = uHigh.size(1);
		CUMAT_KERNEL_3D_LOOP(j, i, b, virtual_size) //virtual_size: W x H x B
		{
			if (maskHigh[b][i][j])
			{
				for (int c = 0; c < C; ++c)
					uHigh[b][c][i][j] = scalar_t(0);
			}
			else
			{
				//interpolate
				const int iLow = i >> 1;
				const int jLow = j >> 1;
				for (int c = 0; c < C; ++c)
				{
					scalar_t v;
					if ((i & 1) == 0 && (j & 1) == 0)
						v = uLow[b][c][iLow][jLow];
					else if ((i & 1) == 1 && (j & 1) == 0)
						v = 0.5 * (uLow[b][c][iLow][jLow] + uLow[b][c][iLow + 1][jLow]);
					else if ((i & 1) == 0 && (j & 1) == 1)
						v = 0.5 * (uLow[b][c][iLow][jLow] + uLow[b][c][iLow][jLow + 1]);
					else
						v = 0.25 * (uLow[b][c][iLow][jLow] + uLow[b][c][iLow + 1][jLow]
							+ uLow[b][c][iLow][jLow + 1] + uLow[b][c][iLow][jLow + 1]);
					uHigh[b][c][i][j] = v;
				}
			}
		}
		CUMAT_KERNEL_3D_LOOP_END
	}

	struct MultigridMemory
	{
		torch::Tensor uHigh[2]; //for pre- and post-smoothing
		torch::Tensor fLow, maskLow;
		torch::Tensor vCoarse[2];

		int init(int sizeHigh, int B, int C, const at::TensorOptions& options)
		{
			int sizeLow = ((sizeHigh - 1) >> 1) + 1;
			uHigh[0] = torch::empty({ B, C, sizeHigh, sizeHigh }, options);
			uHigh[1] = torch::empty({ B, C, sizeHigh, sizeHigh }, options);
			fLow = torch::empty({ B, C, sizeLow, sizeLow }, options);
			maskLow = torch::empty({ B, sizeLow, sizeLow }, options.dtype(c10::kBool));
			vCoarse[0] = torch::empty({ B, C, sizeLow, sizeLow }, options);
			vCoarse[1] = torch::empty({ B, C, sizeLow, sizeLow }, options);
			return sizeLow;
		}
	};

	template<typename scalar_t>
	void multigrid_recursion(
		MultigridMemory* memory, int level,
		const torch::Tensor& u, const torch::Tensor& f, 
		const torch::Tensor& mask, torch::Tensor& uOut,
		scalar_t h,
		int m1, int m2,
		int mc, int ms, int m3,
		cudaStream_t stream)
	{
		const int B = u.size(0);
		const int C = u.size(1);
		const int H = u.size(2);
		const int W = u.size(3);
		const int size = u.size(2);
		const int sizeLow = ((size - 1) >> 1) + 1;

		if (size <= ms)
		{
			//solve coarse grid per-block
			int sharedMemSize = size * size * C * sizeof(scalar_t);
			dim3 block_count = dim3( 1, 1, 1/*B*/ );
			dim3 threads_per_block = dim3( size, size, 1 );
			for (int b=0; b<B; ++b)
				MultigridSmoothBlock <<< block_count, threads_per_block, sharedMemSize, stream >>> (
					u.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					f.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
					h, m3, b,
					uOut.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
			CUMAT_CHECK_ERROR();
			//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			//uOut.copy_(u);
			return;
		}

		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg;
		
		//pre-smooth
		memory[level].uHigh[0].copy_(u);
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		cfg = ctx.createLaunchConfig3D(
			W, H, B, MultigridSmooth<scalar_t>);
		for (int i=0; i<m1; ++i)
		{
			MultigridSmooth<scalar_t>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size,
					memory[level].uHigh[i % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					f.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
					h,
					memory[level].uHigh[(i+1) % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
			CUMAT_CHECK_ERROR();
			//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		}
		//uOut.copy_(memory[level].uHigh[m1 % 2]);

		/*
		 * TODO: the multigrid hierarchy does not converge.
		 * Is something wrong with how the residual is computed and restricted?
		 * The Jacobi smoothing works
		 */
		
		//compute residual
		cfg = ctx.createLaunchConfig3D(
			W, H, B, MultigridResidual<scalar_t>);
		MultigridResidual<scalar_t>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				memory[level].uHigh[m1 % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				f.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
				h,
				memory[level].uHigh[(m1+1) % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()); //residual
		CUMAT_CHECK_ERROR();
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());

		//restrict to coarse grid
		cfg = ctx.createLaunchConfig3D(
			sizeLow, sizeLow, B, MultigridRestrict<scalar_t>);
		MultigridRestrict<scalar_t>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				memory[level].uHigh[(m1+1) % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(), //residual
				f.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
				memory[level].fLow.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				memory[level].maskLow.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>());
		CUMAT_CHECK_ERROR();
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		scalar_t hCoarse = h * 2;

		//recursion
		memory[level].vCoarse[0].fill_(scalar_t(0));
		for (int i = 0; i < mc; ++i)
			multigrid_recursion(
				memory, level + 1,
				memory[level].vCoarse[i % 2],
				memory[level].fLow,
				memory[level].maskLow,
				memory[level].vCoarse[(i + 1) % 2],
				hCoarse,
				m1, m2, mc, ms, m3, stream);
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());

		//interpolate
		cfg = ctx.createLaunchConfig3D(
			size, size, B, MultigridInterpolate<scalar_t>);
		MultigridInterpolate<scalar_t>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				memory[level].vCoarse[mc % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
				memory[level].uHigh[(m1 + 1) % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
		CUMAT_CHECK_ERROR();
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());

		//correct uNew = u+v
		//uNew=u = uHigh[m1%2], v=uHigh[(m1+1)%2]
		memory[level].uHigh[m1 % 2] += memory[level].uHigh[(m1 + 1) % 2];
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());

		//post-smooth
		cfg = ctx.createLaunchConfig3D(
			W, H, B, MultigridSmooth<scalar_t>);
		for (int i=0; i<m2; ++i)
		{
			MultigridSmooth<scalar_t>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size,
					memory[level].uHigh[(m1 + i) % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					f.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
					h,
					memory[level].uHigh[(m1 + i + 1) % 2].packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
			CUMAT_CHECK_ERROR();
			//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		}

		//write output
		uOut.copy_(memory[level].uHigh[(m1 + m2) % 2]);
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	}

	template<typename scalar_t>
	torch::Tensor multigrid(
		const torch::Tensor& mask,
		const torch::Tensor& data,
		int& iterations, float& residual,
		int m0, scalar_t epsilon,
		int m1, int m2,
		int mc, int ms, int m3)
	{
		int64_t B = data.size(0);
		int64_t C = data.size(1);
		int64_t H = data.size(2);
		int64_t W = data.size(3);

		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		
		//compute next power-of-two size
		int size = next_power_of_two(std::max(H, W) - 1) + 1;
		const int originalSize = size;
		scalar_t h = scalar_t(1) / (size - 1);

		//pad to size
		torch::Tensor dataPadded[2];
		auto maskPadded = torch::constant_pad_nd(mask, { 0, size - W, 0, size - H }, 0);
		auto fPadded = torch::constant_pad_nd(data, { 0, size - W, 0, size - H }, scalar_t(0));
		//initialize with fast inpainting
		//dataPadded[0] = torch::zeros_like(fPadded);
		dataPadded[0] = renderer::Inpainting::fastInpaint(maskPadded, fPadded);
		//convert mask
		maskPadded = maskPadded > 0.5;
		//temporar storage
		dataPadded[1] = torch::empty_like(fPadded);
		auto residualPadded = torch::empty_like(fPadded);

		if (ms < 0)
			ms = 17; //largest grid that can be solved in-block
		
		//allocate memory
		std::vector<MultigridMemory> memory;
		at::TensorOptions options = data.options();
		do
		{
			memory.push_back(MultigridMemory());
			size = memory[memory.size() - 1].init(size, B, C, options);
		} while (size > ms);
		
		//run multigrid
		for (iterations=0; iterations < m0; ++iterations)
		{
			//do one multigrid iteration
			multigrid_recursion(memory.data(), 0,
				dataPadded[iterations % 2], fPadded, maskPadded, dataPadded[(iterations + 1) % 2],
				h, m1, m2, mc, ms, m3, stream);
			
			//compute residual norm
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
				originalSize, originalSize, B, MultigridResidual<scalar_t>);
			MultigridResidual<scalar_t>
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size,
					dataPadded[(iterations + 1) % 2].template packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					fPadded.template packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
					maskPadded.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
					h,
					residualPadded.template packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()); //residual
			CUMAT_CHECK_ERROR();
			//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			residual = static_cast<float>(torch::frobenius_norm(residualPadded).template item<scalar_t>());
			//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			//std::cout << "Iteration " << (iterations + 1) << " -> error is " << residual << std::endl;
			if (residual < epsilon) break;
		}

		//remove padding
		return torch::constant_pad_nd(dataPadded[m0 % 2], { 0, -(originalSize - W), 0, -(originalSize - H) }, scalar_t(0));
	}
}

torch::Tensor renderer::Inpainting::pdeInpaint(
	const torch::Tensor& mask,
	const torch::Tensor& data,
	int& iterations, float& residual,
	int64_t m0, double epsilon,
	int64_t m1, int64_t m2,
	int64_t mc, int64_t ms, int64_t m3)
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
	TORCH_CHECK(C < 16, "Inpainting::pdeInpaint only supports up to 16 channels, but got " + std::to_string(C));

	//launch for each type
	torch::Tensor output;
	AT_DISPATCH_FLOATING_TYPES(data.type(), "PDE_Inpainting", ([&]
	{
		output = multigrid<scalar_t>(mask, data, iterations, residual, m0, epsilon, m1, m2, mc, ms, m3);
	}));
	return output;
}

#endif
