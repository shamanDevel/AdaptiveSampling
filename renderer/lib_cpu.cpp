#include "lib.h"

#include <torch/all.h>

torch::Tensor custom_add_cpu(torch::Tensor a, torch::Tensor b)
{
	return a + b;
}