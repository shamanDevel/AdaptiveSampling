#include "hsv_normalization.h"

#include <streambuf>
#include <istream>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ui);

struct membuf : std::streambuf {
	membuf(char const* base, size_t size) {
		char* p(const_cast<char*>(base));
		this->setg(p, p, p + size);
	}
};
struct imemstream : virtual membuf, std::istream {
	imemstream(char const* base, size_t size)
		: membuf(base, size)
		, std::istream(static_cast<std::streambuf*>(this)) {
	}
};

HsvNormalization::HsvNormalization()
{
	//load hsv conversion networks
	auto fs = cmrc::ui::get_filesystem();
	try
	{
		{
			auto ptFile = fs.open("resources/rgb2hsv.pt");
			std::ofstream out("rgb2hsv.pt", std::ofstream::binary | std::ofstream::trunc);
			out.write(ptFile.begin(), ptFile.size());
			out.close();
		}
		//directly loading the script from memory failed with
		// "istream reader failed: seeking to end"
		torch::jit::script::ExtraFilesMap extra_files;
		rgb2hsv_ = torch::jit::load("rgb2hsv.pt", c10::kCUDA, extra_files);
	}
	catch (std::exception& ex)
	{
		std::cerr << "Unable to load rgb2hsv script: " << ex.what() << std::endl;
		throw ex;
	}

	try
	{
		{
			auto ptFile = fs.open("resources/hsv2rgb.pt");
			std::ofstream out("hsv2rgb.pt", std::ofstream::binary | std::ofstream::trunc);
			out.write(ptFile.begin(), ptFile.size());
			out.close();
		}
		//directly loading the script from memory failed with
		// "istream reader failed: seeking to end"
		torch::jit::script::ExtraFilesMap extra_files;
		hsv2rgb_ = torch::jit::load("hsv2rgb.pt", c10::kCUDA, extra_files);
	}
	catch (std::exception& ex)
	{
		std::cerr << "Unable to load hsv2rgb script: " << ex.what() << std::endl;
		throw ex;
	}

	std::cout << "RGB<->HSV conversion scripts loaded" << std::endl;
}

const HsvNormalization * HsvNormalization::Instance()
{
	static HsvNormalization INSTANCE;
	return &INSTANCE;
}

torch::Tensor HsvNormalization::rgb2hsv(const torch::Tensor& rgb) const
{
	TORCH_CHECK(rgb.is_cuda(), "input tensor must reside in CUDA memory");
	TORCH_CHECK(rgb.dim() >= 3, "Tensor must be of shape (*, 3, H, W), but got ", rgb.sizes());
	TORCH_CHECK(rgb.size(rgb.dim() - 3) == 3, "input tensor must be of shape (*, 3, H, W), but got ", rgb.sizes());
	TORCH_CHECK(rgb.dtype() == at::kFloat, "Expected float tensor, but got ", rgb.dtype());
	
	return rgb2hsv_.forward({ rgb }).toTensor();
}

torch::Tensor HsvNormalization::hsv2rgb(const torch::Tensor & hsv) const
{
	TORCH_CHECK(hsv.is_cuda(), "input tensor must reside in CUDA memory");
	TORCH_CHECK(hsv.dim() >= 3, "Tensor must be of shape (*, 3, H, W), but got ", hsv.sizes());
	TORCH_CHECK(hsv.size(hsv.dim() - 3) == 3, "input tensor must be of shape (*, 3, H, W), but got ", hsv.sizes());
	TORCH_CHECK(hsv.dtype() == at::kFloat, "Expected float tensor, but got ", hsv.dtype());
	
	return hsv2rgb_.forward({ hsv }).toTensor();
}

torch::Tensor HsvNormalization::normalizeHsvValue(const torch::Tensor& in, float& scalingOut) const
{
	torch::Tensor hsv = rgb2hsv(in);
	//compute scaling factor
	float max_value = torch::max(hsv.narrow(hsv.dim() - 3, 2, 1)).item<float>();
	scalingOut = 1.0f / max_value;
	//do the scaling
	hsv.narrow(hsv.dim() - 3, 2, 1) *= scalingOut;
	//convert back
	return hsv2rgb(hsv);
}

torch::Tensor HsvNormalization::denormalizeHsvValue(const torch::Tensor& in, float scalingIn) const
{
	torch::Tensor hsv = rgb2hsv(in);
	hsv.narrow(hsv.dim() - 3, 2, 1) *= 1.0f / scalingIn;
	return hsv2rgb(hsv);
}
