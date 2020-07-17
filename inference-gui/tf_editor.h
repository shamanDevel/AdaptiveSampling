#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <deque>
#include <vector>

#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "tf_texture_1d.h"
#include <json.hpp>

class TfEditorOpacity
{
public:
	TfEditorOpacity();
	void init(const ImRect& rect);
	void updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float>& opacityAxis);
	void handleIO();
	void render();

	const std::vector<float>& getDensityAxis() const { return densityAxis_; }
	const std::vector<float>& getOpacityAxis() const { return opacityAxis_; }
	bool getIsChanged() const { return isChanged_; }

private:
	const float circleRadius_{ 4.0f };

	ImRect tfEditorRect_;
	int selectedControlPoint_{ -1 };
	std::deque<ImVec2> controlPoints_;
	std::vector<float> densityAxis_;
	std::vector<float> opacityAxis_;

	bool isChanged_{ false };

private:
	ImRect createControlPointRect(const ImVec2& controlPoint);
	ImVec2 screenToEditor(const ImVec2& screenPosition);
	ImVec2 editorToScreen(const ImVec2& editorPosition);
};

class TfEditorColor
{
public:
	//Non-copyable and non-movable
	TfEditorColor();
	~TfEditorColor();
	TfEditorColor(const TfEditorColor&) = delete;
	TfEditorColor(TfEditorColor&&) = delete;

	void init(const ImRect& rect, bool showControlPoints);
	void updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float3>& colorAxis);
	void handleIO();
	void render();

	const std::vector<float>& getDensityAxis() const { return densityAxis_; }
	const std::vector<float3>& getColorAxis() const { return colorAxis_; }
	bool getIsChanged() const { return isChanged_; }

private:
	const float cpWidth_{ 8.0f };

	ImVec4 pickedColor_{ 0.0f, 0.0f, 1.0f, 1.0f };
	ImRect tfEditorRect_;
	int selectedControlPointForMove_{ -1 };
	int selectedControlPointForColor_{ -1 };
	std::deque<ImVec4> controlPoints_;
	std::vector<float> densityAxis_;
	std::vector<float3> colorAxis_;
	bool isChanged_{ false };
	bool showControlPoints_{ true };

	//Variables for color map texture.
	cudaGraphicsResource* resource_{ nullptr };
	GLuint colorMapImage_{ 0 };
	cudaSurfaceObject_t content_{ 0 };
	cudaArray_t contentArray_{ nullptr };
	RENDERER_NAMESPACE TfTexture1D tfTexture_;

private:
	void destroy();
	ImRect createControlPointRect(float x);
	float screenToEditor(float screenPositionX);
	float editorToScreen(float editorPositionX);
};

class TfEditor
{
public:
	void init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints);
	void handleIO();
	void render();
	void saveToFile(const std::string& path, float minDensity, float maxDensity) const;
	void loadFromFile(const std::string& path, float& minDensity, float& maxDensity);
	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

	const std::vector<float>& getDensityAxisOpacity() const { return editorOpacity_.getDensityAxis(); }
	const std::vector<float>& getOpacityAxis() const { return editorOpacity_.getOpacityAxis(); }
	const std::vector<float>& getDensityAxisColor() const { return editorColor_.getDensityAxis(); }
	const std::vector<float3>& getColorAxis() const { return editorColor_.getColorAxis(); }
	bool getIsChanged() const { return editorOpacity_.getIsChanged() || editorColor_.getIsChanged(); }

	static bool testIntersectionRectPoint(const ImRect& rect, const ImVec2& point);

private:
	TfEditorOpacity editorOpacity_;
	TfEditorColor editorColor_;
};