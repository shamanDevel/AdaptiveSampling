#pragma once

#include <torch/types.h>

#ifdef BUILD_MAIN_LIB
#define MY_API C10_EXPORT
#else
#define MY_API C10_IMPORT
#endif

#define BEGIN_RENDERER_NAMESPACE namespace renderer {
#define END_RENDERER_NAMESPACE }
#define RENDERER_NAMESPACE ::renderer::
