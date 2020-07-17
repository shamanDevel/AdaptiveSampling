#include "camera.h"

#include <iostream>
#include <iomanip>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtc/matrix_access.hpp"
#include "glm/gtc/matrix_transform.hpp"

#ifdef RENDERER_HAS_RENDERER

namespace std
{
	std::ostream& operator<<(std::ostream& o, const glm::vec3 v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::vec4 v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z << "," << v.w;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::mat4 m)
	{
		o << m[0] << "\n" << m[1] << "\n" << m[2] << "\n" << m[3];
		return o;
	}
}

BEGIN_RENDERER_NAMESPACE

//Source: Cinder, Matrix.cpp
static glm::mat4 alignZAxisWithTarget(glm::vec3 targetDir, glm::vec3 upDir)
{
	// Ensure that the target direction is non-zero.
	if (length2(targetDir) == 0)
		targetDir = glm::vec3(0, 0, 1);

	// Ensure that the up direction is non-zero.
	if (length2(upDir) == 0)
		upDir = glm::vec3(0, 1, 0);

	// Check for degeneracies.  If the upDir and targetDir are parallel 
	// or opposite, then compute a new, arbitrary up direction that is
	// not parallel or opposite to the targetDir.
	if (length2(cross(upDir, targetDir)) == 0) {
		upDir = cross(targetDir, glm::vec3(1, 0, 0));
		if (length2(upDir) == 0)
			upDir = cross(targetDir, glm::vec3(0, 0, 1));
	}

	// Compute the x-, y-, and z-axis vectors of the new coordinate system.
	glm::vec3 targetPerpDir = cross(upDir, targetDir);
	glm::vec3 targetUpDir = cross(targetDir, targetPerpDir);

	// Rotate the x-axis into targetPerpDir (row 0),
	// rotate the y-axis into targetUpDir   (row 1),
	// rotate the z-axis into targetDir     (row 2).
	glm::vec3 row[3];
	row[0] = normalize(targetPerpDir);
	row[1] = normalize(targetUpDir);
	row[2] = normalize(targetDir);

	const float v[16] = { row[0].x,  row[0].y,  row[0].z,  0,
							row[1].x,  row[1].y,  row[1].z,  0,
							row[2].x,  row[2].y,  row[2].z,  0,
							0,         0,         0,		 1 };

	glm::mat4 result;
	memcpy(&result[0].x, v, sizeof(glm::mat4));
	return result;
}

namespace{
	// copy of glm::perspectiveFovLH_ZO, seems to not be defined in unix
	glm::mat4 perspectiveFovLH_ZO(float fov, float width, float height, float zNear, float zFar)
	{
		assert(width > static_cast<float>(0));
		assert(height > static_cast<float>(0));
		assert(fov > static_cast<float>(0));

		float const rad = fov;
		float const h = glm::cos(static_cast<float>(0.5) * rad) / glm::sin(static_cast<float>(0.5) * rad);
		float const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		glm::mat4 Result(static_cast<float>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = zFar / (zFar - zNear);
		Result[2][3] = static_cast<float>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}
}

void Camera::computeMatrices(float3 cameraOrigin_, float3 cameraLookAt_, float3 cameraUp_, float fovDegrees,
	int width, int height, float nearClip, float farClip, float4 viewMatrixOut[4], float4 viewMatrixInverseOut[4],
	float4 normalMatrixOut[4])
{
	const glm::vec3 cameraOrigin = *reinterpret_cast<glm::vec3*>(&cameraOrigin_.x);
	const glm::vec3 cameraLookAt = *reinterpret_cast<glm::vec3*>(&cameraLookAt_.x);
	const glm::vec3 cameraUp = *reinterpret_cast<glm::vec3*>(&cameraUp_.x);
	//std::cout << "cameraOrigin: " << cameraOrigin << std::endl;
	//std::cout << "cameraLookAt: " << cameraLookAt << std::endl;
	//std::cout << "cameraUp: " << cameraUp << std::endl;
	//std::cout << "clip near=" << nearClip << ", far=" << farClip << std::endl;

	float fovRadians = glm::radians(fovDegrees);
	//std::cout << "fov: " << fovDegrees << "° = " << fovRadians << "rad" << std::endl;

#if 1
	glm::mat4 viewMatrix = glm::lookAtLH(cameraOrigin, cameraLookAt, normalize(cameraUp));
	glm::mat4 projMatrix = perspectiveFovLH_ZO(fovRadians, float(width), float(height), nearClip, farClip);
#else
	//computation code from Cinder
	float aspectRatio = width / float(height);
	float mFrustumTop = nearClip * std::tan(0.5f * fovRadians);
	float mFrustumBottom = -mFrustumTop;
	float mFrustumRight = mFrustumTop * aspectRatio;
	float mFrustumLeft = -mFrustumRight;
	std::cout << "frustum top:    " << mFrustumTop << std::endl;
	std::cout << "frustum bottom: " << mFrustumBottom << std::endl;
	std::cout << "frustum right:  " << mFrustumRight << std::endl;
	std::cout << "frustum left:   " << mFrustumLeft << std::endl;

	glm::mat4 projMatrix(0);
	projMatrix[0][0] = 2.0f * nearClip / (mFrustumRight - mFrustumLeft);
	projMatrix[1][0] = 0.0f;
	projMatrix[2][0] = (mFrustumRight + mFrustumLeft) / (mFrustumRight - mFrustumLeft);
	projMatrix[3][0] = 0.0f;
	projMatrix[0][1] = 0.0f;
	projMatrix[1][1] = 2.0f * nearClip / (mFrustumTop - mFrustumBottom);
	projMatrix[2][1] = (mFrustumTop + mFrustumBottom) / (mFrustumTop - mFrustumBottom);
	projMatrix[3][1] = 0.0f;
	projMatrix[0][2] = 0.0f;
	projMatrix[1][2] = 0.0f;
	projMatrix[2][2] = -(farClip + nearClip) / (farClip - nearClip);
	projMatrix[3][2] = -2.0f * farClip * nearClip / (farClip - nearClip);
	projMatrix[0][3] = 0.0f;
	projMatrix[1][3] = 0.0f;
	projMatrix[2][3] = -1.0f;
	projMatrix[3][3] = 0.0f;
	projMatrix = glm::transpose(projMatrix);

	glm::vec3 viewDirection = glm::normalize(cameraLookAt - cameraOrigin);
	glm::mat4 alignMat = alignZAxisWithTarget(-viewDirection, cameraUp);
	glm::quat orientation = glm::toQuat(alignMat);
	glm::vec3 mW = -normalize(viewDirection);
	glm::vec3 mU = glm::rotate(orientation, glm::vec3(1, 0, 0));
	glm::vec3 mV = glm::rotate(orientation, glm::vec3(0, 1, 0));
	glm::vec3 d(-dot(cameraOrigin, mU), -dot(cameraOrigin, mV), -dot(cameraOrigin, mW));
	std::cout << "Align-Matrix:\n" << alignMat << std::endl;
	std::cout << "orientation: " << orientation.x << "," << orientation.y << "," << orientation.z << "," << orientation.w << std::endl;
	std::cout << "mW: " << mW << std::endl;
	std::cout << "mU: " << mU << std::endl;
	std::cout << "mV: " << mV << std::endl;
	glm::mat4 viewMatrix(0);
	viewMatrix[0][0] = mU.x; viewMatrix[1][0] = mU.y; viewMatrix[2][0] = mU.z; viewMatrix[3][0] = d.x;
	viewMatrix[0][1] = mV.x; viewMatrix[1][1] = mV.y; viewMatrix[2][1] = mV.z; viewMatrix[3][1] = d.y;
	viewMatrix[0][2] = mW.x; viewMatrix[1][2] = mW.y; viewMatrix[2][2] = mW.z; viewMatrix[3][2] = d.z;
	viewMatrix[0][3] = 0.0f; viewMatrix[1][3] = 0.0f; viewMatrix[2][3] = 0.0f; viewMatrix[3][3] = 1.0f;
#endif

	glm::mat4 viewProjMatrix = projMatrix * viewMatrix;
	glm::mat4 invViewProjMatrix = glm::inverse(viewProjMatrix);
	glm::mat4 normalMatrix = glm::inverse(glm::transpose(glm::mat4(glm::mat3(viewMatrix))));

	//std::cout << "projMatrix:\n" << projMatrix << std::endl;
	//std::cout << "viewMatrix:\n" << viewMatrix << std::endl;
	//std::cout << "viewProjMatrix:\n" << viewProjMatrix << std::endl;
	//std::cout << "invViewProjMatrix:\n" << invViewProjMatrix << std::endl;
	//std::cout << "normalMatrix:\n" << normalMatrix << std::endl;

	//auto printRay = [&](int x, int y)
	//{
	//	glm::vec4 vec(x, y, 1, 1);
	//	glm::vec4 res = invViewProjMatrix * vec;
	//	res /= res.w;
	//	glm::vec3 dir = glm::normalize(glm::vec3(res) - cameraOrigin);
	//	glm::vec3 normal = -dir;
	//	normal = glm::vec3(normalMatrix * glm::vec4(normal, 0));
	//	std::cout << "[" << x << ", " << y << "] -> (" << dir.x << ", " << dir.y << ", " << dir.z << ")" 
	//		<< " n=(" << normal.x << ", " << normal.y << ", " << normal.z << ")" << std::endl;
	//};
	//printRay(-1, -1);
	//printRay(-1, +1);
	//printRay(+1, -1);
	//printRay(+1, +1);
	//printRay(0, 0);

	viewProjMatrix = glm::transpose(viewProjMatrix);
	invViewProjMatrix = glm::transpose(invViewProjMatrix);
	normalMatrix = glm::transpose(normalMatrix);
	normalMatrix[0] = -normalMatrix[0]; //somehow, the networks were trained with normal-x inverted
	for (int i = 0; i < 4; ++i) viewMatrixOut[i] = *reinterpret_cast<float4*>(&viewProjMatrix[i].x);
	for (int i = 0; i < 4; ++i) viewMatrixInverseOut[i] = *reinterpret_cast<float4*>(&invViewProjMatrix[i].x);
	for (int i = 0; i < 4; ++i) normalMatrixOut[i] = *reinterpret_cast<float4*>(&normalMatrix[i].x);
}

END_RENDERER_NAMESPACE
#endif
