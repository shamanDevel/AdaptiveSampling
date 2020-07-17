#include "lib.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor custom_add_gpu(torch::Tensor a, torch::Tensor b)
{
	CHECK_INPUT(a);
	CHECK_INPUT(b);
	std::cout << "Test" << std::endl;
	torch::Tensor foo = a + b + 1;
	std::cout << "Done" << std::endl;
	return foo;
	//return a + b;
}