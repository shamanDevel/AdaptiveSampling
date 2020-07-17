#include <torch/script.h>

#include "lib.h"

#define TORCH_EXTENSION_NAME "renderer"

using namespace renderer;

static auto registry = torch::RegisterOperators()
#ifdef RENDERER_HAS_RENDERER
.op(TORCH_EXTENSION_NAME "::close_volume", &CloseVolume)
.op(TORCH_EXTENSION_NAME "::load_volume_from_raw", &LoadVolumeFromRaw)
.op(TORCH_EXTENSION_NAME "::load_volume_from_xyz", &LoadVolumeFromXYZ)
.op(TORCH_EXTENSION_NAME "::load_volume_from_binary", &LoadVolumeFromBinary)
.op(TORCH_EXTENSION_NAME "::get_volume_resolution", &GetVolumeResolution)
.op(TORCH_EXTENSION_NAME "::save_volume_to_binary", &SaveVolumeToBinary)
.op(TORCH_EXTENSION_NAME "::create_mipmap_level", &CreateMipmapLevel)
.op(TORCH_EXTENSION_NAME "::create_volume_from_tensor", &CreateVolumeFromPytorch)
.op(TORCH_EXTENSION_NAME "::volume_to_tensor", &ConvertVolumeToPytorch)
.op(TORCH_EXTENSION_NAME "::cast_volume", &ConvertVolumeToDtype)
.op(TORCH_EXTENSION_NAME "::get_histogram", &GetHistogram)
.op(TORCH_EXTENSION_NAME "::set_renderer_parameter", &SetRendererParameter)
.op(TORCH_EXTENSION_NAME "::initialize_renderer", &initializeRenderer)
.op(TORCH_EXTENSION_NAME "::render", &Render)
.op(TORCH_EXTENSION_NAME "::render_samples", &RenderSamples)
.op(TORCH_EXTENSION_NAME "::render_adaptive_stepsize", &RenderAdaptiveStepsize)
.op(TORCH_EXTENSION_NAME "::scatter_samples", &ScatterSamplesToImage)
.op(TORCH_EXTENSION_NAME "::regular_interp1d", &RegularInterp1d)
#endif
#ifdef RENDERER_HAS_SAMPLING
.op(TORCH_EXTENSION_NAME "::load_samples", &loadSamplesFromFile)
.op(TORCH_EXTENSION_NAME "::compute_average_sample_distance", &computeAverageSampleDistance)
.op(TORCH_EXTENSION_NAME "::adaptive_smoothing", &adaptiveSmoothing)
.op(TORCH_EXTENSION_NAME "::adjoint_adaptive_smoothing", &adjAdaptiveSmoothing)
#endif
#ifdef RENDERER_HAS_MESH_DRAWER
.op(TORCH_EXTENSION_NAME "::create_offscreen_context", &CreateOffscreenContext)
.op(TORCH_EXTENSION_NAME "::create_mesh_drawer", &CreateMeshDrawer)
.op(TORCH_EXTENSION_NAME "::render_mesh", &RenderMesh)
#endif
#ifdef RENDERER_HAS_INPAINTING
.op(TORCH_EXTENSION_NAME "::fast_inpaint", &Inpainting::fastInpaint)
.op(TORCH_EXTENSION_NAME "::fast_inpaint_fractional", &Inpainting::fastInpaintFractional)
.op(TORCH_EXTENSION_NAME "::adj_fast_inpaint_fractional", &Inpainting::adjFastInpaintingFractional)
.op(TORCH_EXTENSION_NAME "::pde_inpaint", [](
	const torch::Tensor& mask,
	const torch::Tensor& data,
	int64_t m0, double epsilon,
	int64_t m1, int64_t m2,
	int64_t mc, int64_t ms, int64_t m3)
{
	int iterations; float residual;
	return Inpainting::pdeInpaint(mask, data, iterations, residual,
		m0, epsilon, m1, m2, mc, ms, m3);
})
#endif
;
//.op(xstr(TORCH_EXTENSION_NAME) "::add_cpu", &custom_add_cpu)
//.op(xstr(TORCH_EXTENSION_NAME) "::add_gpu", &custom_add_gpu);