#pragma once

#include "commons.h"

#include <cuda_runtime.h>

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

class MY_API Camera
{
public:

	/**
	 * \brief Computes the perspective camera matrices
	 * \param cameraOrigin camera origin / eye pos
	 * \param cameraLookAt look at / target
	 * \param cameraUp up vector
	 * \param fovDegrees vertical field-of-views in degree
	 * \param width the width of the screen
	 * \param height the height of the screen
	 * \param nearClip the near clipping plane
	 * \param farClip the far clipping plane
	 * \param viewMatrixOut view-projection matrix in Row Major order (viewMatrixOut[0] is the first row), [OUT]
	 * \param viewMatrixInverseOut inverse view-projection matrix in Row Major order (viewMatrixInverseOut[0] is the first row), [OUT]
	 * \param normalMatrixOut normal matrix in Row Major order (normalMatrixOut[0] is the first row), [OUT]
	 */
	static void computeMatrices(
		float3 cameraOrigin, float3 cameraLookAt, float3 cameraUp, 
		float fovDegrees, int width, int height, float nearClip, float farClip,
		float4 viewMatrixOut[4], float4 viewMatrixInverseOut[4], float4 normalMatrixOut[4]
	);
};

END_RENDERER_NAMESPACE
#endif
