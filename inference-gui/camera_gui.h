#pragma once

#include <lib.h>
#include <json.hpp>

class CameraGui
{
public:
	/**
	 * \brief Specifies the UI
	 * \return true if properties have changed
	 */
	bool specifyUI();

	/**
	 * \brief Updates movement with mouse dragging
	 * and scroll wheel zooming
	 * \return true if properties have changed
	 */
	bool updateMouse();

	/**
	 * \brief Sets cameraOrigin, cameraLookAt,
	 *  cameraUp and cameraFovDegrees.
	 */
	void updateRenderArgs(renderer::RendererArgs& args) const;

	/**
	 * \brief Converts a direction in screen space (X,Y,Z) to
	 * the direction in world space (right,up,viewDir).
	 * \param screenDirection the screen space direction
	 */
	float3 screenToWorld(const float3& screenDirection) const;

	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

private:
	void computeParameters(
		float3& origin, float3& up, float& distance) const;

	enum Orientation
	{
		Xp, Xm, Yp, Ym, Zp, Zm
	};
	static const char* OrientationNames[6];
	static const float3 OrientationUp[6];
	static const int3 OrientationPermutation[6];
	static const bool OrientationInvertYaw[6];
	static const bool OrientationInvertPitch[6];
	Orientation orientation_ = Ym;

	float3 lookAt_{0,0,0};
	float rotateSpeed_ = 0.5f;
	float zoomSpeed_ = 1.1;

	float fov_ = 45.0f;
	const float baseDistance_ = 1.0f;
	float currentPitch_ = 67.0f;
	float currentYaw_ = 96.0f;
	float zoomvalue_ = 0;

	float mouseWheel_ = 0;
};