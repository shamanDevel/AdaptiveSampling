#include "regular_interp1d.h"

#include <algorithm>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

#define CLAMP(x, a, b) \
	(max(min((x), (b)), (a)))
#define INTERP(frac, a, b) \
	((a) + (frac)*((b)-(a)))

template<typename scalar_t>
static void eval_cpu(
	const torch::Tensor& heights, const torch::Tensor& newX, const torch::Tensor& out)
{
	int64_t B = heights.size(0);
	int64_t N = heights.size(1);
	int64_t M = newX.size(1);

	const auto H = heights.accessor<scalar_t, 2>();
	const auto X = newX.accessor<scalar_t, 2>();
	auto O = out.accessor<scalar_t, 2>();

	using std::min;
	using std::max;
#pragma omp parallel for
	for (int64_t b = 0; b<B; ++b)
		for (int64_t i = 0; i<M; ++i)
		{
			scalar_t x = X[b][i];
			//convert to bin index
			x *= N - 1;
			int64_t binIndex = int64_t(x);
			scalar_t binFrac = x - binIndex;
			//get sample points
			scalar_t ha = H[b][CLAMP(binIndex, 0ll, N)];
			scalar_t hb = H[b][CLAMP(binIndex+1, 0ll, N)];
			//write output
			O[b][i] = INTERP(binFrac, ha, hb);
		}
}

template<typename scalar_t>
static void eval_gpu(
	const torch::Tensor& heights, const torch::Tensor& newX, const torch::Tensor& out)
{
	throw std::exception("not implemented yet");
}

void RegularInterp1d(torch::Tensor heights, torch::Tensor newX, torch::Tensor output)
{
	AT_ASSERTM(heights.dim() == 2, "heights must be a 2D tensor");
	AT_ASSERTM(newX.dim() == 2, "newX must be a 2D tensor");
	AT_ASSERTM(heights.size(0) == newX.size(0), "batch dimension must agree");
	AT_ASSERTM(heights.device() == newX.device(), "heights and newX must reside on the same device");
	AT_ASSERTM(heights.dtype() == newX.dtype(), "data types of heights and newX must agree");
	
	AT_ASSERTM(output.dim() == 2, "output must be a 2D tensor");
	AT_ASSERTM(output.size(0) == newX.size(0), "batch dimension must agree");
	AT_ASSERTM(output.size(1) == newX.size(1), "number of sample points must agree");
	AT_ASSERTM(output.device() == newX.device(), "output and newX must reside on the same device");
	AT_ASSERTM(output.dtype() == newX.dtype(), "data types of output and newX must agree");

	AT_DISPATCH_FLOATING_TYPES(heights.type(), "regular_interp1d", ([&]
	{
		if (heights.device().is_cuda())
			eval_gpu<scalar_t>(heights, newX, output);
		else
			eval_cpu<scalar_t>(heights, newX, output);
	}));
}


END_RENDERER_NAMESPACE
#endif