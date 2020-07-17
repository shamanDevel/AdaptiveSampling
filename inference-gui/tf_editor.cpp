#include "tf_editor.h"
#include "visualizer_kernels.h"

#include <algorithm>
#include <cuda_gl_interop.h>
#include <cuMat/src/Context.h>
#include "utils.h"

TfEditorOpacity::TfEditorOpacity()
	: controlPoints_({ ImVec2(0.45f, 0.0f), ImVec2(0.5f, 0.8f), ImVec2(0.55f, 0.0f) })
	, densityAxis_({ 0.45f, 0.5f, 0.55f })
	, opacityAxis_({ 0.0f, 0.8f, 0.0f })
{}

void TfEditorOpacity::init(const ImRect& rect)
{
	tfEditorRect_ = rect;
}

void TfEditorOpacity::updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float>& opacityAxis)
{
	assert(densityAxis.size() == opacityAxis.size());
	assert(densityAxis.size() >= 1 && opacityAxis.size() >= 1);

	selectedControlPoint_ = -1;
	controlPoints_.clear();
	densityAxis_ = densityAxis;
	opacityAxis_ = opacityAxis;

	int size = densityAxis.size();
	for (int i = 0; i < size; ++i)
	{
		controlPoints_.emplace_back(densityAxis_[i], opacityAxis_[i]);
	}
}

void TfEditorOpacity::handleIO()
{
	isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	//Early leave if mouse is not on opacity editor and no control point is selected.
	if (!TfEditor::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPoint_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		controlPoints_.push_back(screenToEditor(mousePosition));
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPoint_ >= 0)
		{
			isChanged_ = true;

			ImVec2 center(std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x),
				std::min(std::max(mousePosition.y, tfEditorRect_.Min.y), tfEditorRect_.Max.y));

			controlPoints_[selectedControlPoint_] = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			for (int idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx]));
				if (TfEditor::testIntersectionRectPoint(cp, mousePosition))
				{
					selectedControlPoint_ = idx;
					break;
				}
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		for (int idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx]));
			if (TfEditor::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				selectedControlPoint_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPoint_ = -1;
	}
}

void TfEditorOpacity::render()
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(tfEditorRect_.Min, tfEditorRect_.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const ImVec2& p1, const ImVec2& p2)
		{
			return p1.x < p2.x;
		});

	//Fill densityAxis_ and opacityAxis_ and convert coordinates from editor space to screen space.
	densityAxis_.clear();
	opacityAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		densityAxis_.push_back(cp.x);
		opacityAxis_.push_back(cp.y);
		cp = editorToScreen(cp);
	}

	//Draw lines between the control points.
	int size = controlPointsRender.size();
	for (int i = 0; i < size + 1; ++i)
	{
		auto left = (i == 0) ? ImVec2(tfEditorRect_.Min.x, controlPointsRender.front().y) : controlPointsRender[i - 1];
		auto right = (i == size) ? ImVec2(tfEditorRect_.Max.x, controlPointsRender.back().y) : controlPointsRender[i];

		window->DrawList->AddLine(left, right, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 1.0f);
	}

	//Draw the control points
	for (const auto& cp : controlPointsRender)
	{
		window->DrawList->AddCircleFilled(cp, circleRadius_, ImColor(ImVec4(0.0f, 1.0f, 0.0f, 1.0f)), 16);
	}
}

ImRect TfEditorOpacity::createControlPointRect(const ImVec2& controlPoint)
{
	return ImRect(ImVec2(controlPoint.x - circleRadius_, controlPoint.y - circleRadius_),
		ImVec2(controlPoint.x + circleRadius_, controlPoint.y + circleRadius_));
}

ImVec2 TfEditorOpacity::screenToEditor(const ImVec2& screenPosition)
{
	ImVec2 editorPosition;
	editorPosition.x = (screenPosition.x - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);
	editorPosition.y = 1.0f - (screenPosition.y - tfEditorRect_.Min.y) / (tfEditorRect_.Max.y - tfEditorRect_.Min.y);

	return editorPosition;
}

ImVec2 TfEditorOpacity::editorToScreen(const ImVec2& editorPosition)
{
	ImVec2 screenPosition;
	screenPosition.x = editorPosition.x * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;
	screenPosition.y = (1.0f - editorPosition.y) * (tfEditorRect_.Max.y - tfEditorRect_.Min.y) + tfEditorRect_.Min.y;

	return screenPosition;
}

TfEditorColor::TfEditorColor()
	: densityAxis_({ 0.0f, 1.0f })
{
	auto red = renderer::rgbToLab(make_float3(1.0f, 0.0f, 0.0f));
	auto white = renderer::rgbToLab(make_float3(1.0f, 1.0f, 1.0f));

	controlPoints_.emplace_back(0.0f, red.x, red.y, red.z);
	controlPoints_.emplace_back(1.0f, white.x, white.y, white.z);

	colorAxis_.push_back(red);
	colorAxis_.push_back(white);
}

TfEditorColor::~TfEditorColor()
{
	destroy();
}

void TfEditorColor::init(const ImRect& rect, bool showControlPoints)
{
	ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputHSV;
	ImGui::ColorEdit3("", &pickedColor_.x, colorFlags);

	showControlPoints_ = showControlPoints;

	//If editor is created for the first time or its size is changed, create CUDA texture.
	if (tfEditorRect_.Min.x == FLT_MAX ||
		!(rect.Min.x == tfEditorRect_.Min.x &&
			rect.Min.y == tfEditorRect_.Min.y &&
			rect.Max.x == tfEditorRect_.Max.x &&
			rect.Max.y == tfEditorRect_.Max.y))
	{
		destroy();
		tfEditorRect_ = rect;

		auto colorMapWidth = tfEditorRect_.Max.x - tfEditorRect_.Min.x;
		auto colorMapHeight = tfEditorRect_.Max.y - tfEditorRect_.Min.y;
		glGenTextures(1, &colorMapImage_);

		glBindTexture(GL_TEXTURE_2D, colorMapImage_);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorMapWidth, colorMapHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(&resource_, colorMapImage_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		CUMAT_SAFE_CALL(cudaMallocArray(&contentArray_, &channelDesc, colorMapWidth, colorMapHeight, cudaArraySurfaceLoadStore));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;

		resDesc.res.array.array = contentArray_;
		CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&content_, &resDesc));
	}
}

void TfEditorColor::updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float3>& colorAxis)
{
	selectedControlPointForMove_ = -1;
	selectedControlPointForColor_ = -1;
	controlPoints_.clear();
	densityAxis_ = densityAxis;
	colorAxis_ = colorAxis;

	int size = densityAxis.size();
	for (int i = 0; i < size; ++i)
	{
		controlPoints_.emplace_back(densityAxis_[i], colorAxis_[i].x, colorAxis_[i].y, colorAxis_[i].z);
	}
}

void TfEditorColor::handleIO()
{
	isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	if (selectedControlPointForColor_ >= 0)
	{
		auto& cp = controlPoints_[selectedControlPointForColor_];

		float3 pickedColorLab;
		ImGui::ColorConvertHSVtoRGB(pickedColor_.x, pickedColor_.y, pickedColor_.z, pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
		pickedColorLab = renderer::rgbToLab(pickedColorLab);

		if (cp.y != pickedColorLab.x || cp.z != pickedColorLab.y ||
			cp.w != pickedColorLab.z)
		{
			cp.y = pickedColorLab.x;
			cp.z = pickedColorLab.y;
			cp.w = pickedColorLab.z;
			isChanged_ = true;
		}
	}

	//Early leave if mouse is not on color editor.
	if (!TfEditor::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPointForMove_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		float3 pickedColorLab;
		ImGui::ColorConvertHSVtoRGB(pickedColor_.x, pickedColor_.y, pickedColor_.z, pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
		pickedColorLab = renderer::rgbToLab(pickedColorLab);
		controlPoints_.emplace_back(screenToEditor(mousePosition.x), pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPointForMove_ >= 0)
		{
			isChanged_ = true;

			float center = std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x);

			controlPoints_[selectedControlPointForMove_].x = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			int idx;
			for (idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].x));
				if (TfEditor::testIntersectionRectPoint(cp, mousePosition))
				{
					selectedControlPointForColor_ = selectedControlPointForMove_ = idx;

					auto colorRgb = renderer::labToRgb(make_float3(controlPoints_[selectedControlPointForMove_].y,
						controlPoints_[selectedControlPointForMove_].z,
						controlPoints_[selectedControlPointForMove_].w));

					ImGui::ColorConvertRGBtoHSV(colorRgb.x, colorRgb.y, colorRgb.z, pickedColor_.x, pickedColor_.y, pickedColor_.z);
					break;
				}
			}

			//In case of no hit on any control point, unselect for color pick as well.
			if (idx == size)
			{
				selectedControlPointForColor_ = -1;
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		int idx;
		for (idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].x));
			if (TfEditor::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				selectedControlPointForColor_ = selectedControlPointForMove_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPointForMove_ = -1;
	}
}

void TfEditorColor::render()
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const ImVec4& cp1, const ImVec4& cp2)
		{
			return cp1.x < cp2.x;
		});

	//Fill densityAxis_ and colorAxis_.
	densityAxis_.clear();
	colorAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		densityAxis_.push_back(cp.x);
		colorAxis_.push_back(make_float3(cp.y, cp.z, cp.w));
	}

	auto colorMapWidth = tfEditorRect_.Max.x - tfEditorRect_.Min.x;
	auto colorMapHeight = tfEditorRect_.Max.y - tfEditorRect_.Min.y;

	//Write to color map texture.
	tfTexture_.updateIfChanged({ 0.0f, 1.0f }, { 0.0f, 1.0f }, densityAxis_, colorAxis_);
	kernel::fillColorMap(content_, tfTexture_.getTextureObject(), colorMapWidth, colorMapHeight);

	//Draw color interpolation between control points.
	cudaArray_t texturePtr = nullptr;
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &resource_, 0));
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texturePtr, resource_, 0, 0));
	CUMAT_SAFE_CALL(cudaMemcpyArrayToArray(texturePtr, 0, 0, contentArray_, 0, 0, colorMapWidth * colorMapHeight * 4, cudaMemcpyDeviceToDevice));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource_, 0));

	window->DrawList->AddImage((void*)colorMapImage_, tfEditorRect_.Min, tfEditorRect_.Max);

	if (showControlPoints_)
	{
		//Draw the control points
		int cpIndex = 0;
		for (const auto& cp : controlPoints_)
		{
			//If this is the selected control point, use different color.
			auto rect = createControlPointRect(editorToScreen(cp.x));
			if (selectedControlPointForColor_ == cpIndex++)
			{
				window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(1.0f, 0.8f, 0.1f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 3.0f);
			}
			else
			{
				window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 2.0f);
			}
		}
	}
}

void TfEditorColor::destroy()
{
	if (colorMapImage_)
	{
		glDeleteTextures(1, &colorMapImage_);
		colorMapImage_ = 0;
	}
	if (content_)
	{
		CUMAT_SAFE_CALL(cudaDestroySurfaceObject(content_));
		content_ = 0;
	}
	if (contentArray_)
	{
		CUMAT_SAFE_CALL(cudaFreeArray(contentArray_));
		contentArray_ = nullptr;
	}
}

ImRect TfEditorColor::createControlPointRect(float x)
{
	return ImRect(ImVec2(x - 0.5f * cpWidth_, tfEditorRect_.Min.y),
		ImVec2(x + 0.5f * cpWidth_, tfEditorRect_.Max.y));
}

float TfEditorColor::screenToEditor(float screenPositionX)
{
	float editorPositionX;
	editorPositionX = (screenPositionX - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);

	return editorPositionX;
}

float TfEditorColor::editorToScreen(float editorPositionX)
{
	float screenPositionX;
	screenPositionX = editorPositionX * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;

	return screenPositionX;
}

void TfEditor::init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints)
{
	editorOpacity_.init(tfEditorOpacityRect);
	editorColor_.init(tfEditorColorRect, showColorControlPoints);
}

void TfEditor::handleIO()
{
	editorOpacity_.handleIO();
	editorColor_.handleIO();
}

void TfEditor::render()
{
	editorOpacity_.render();
	editorColor_.render();
}

void TfEditor::saveToFile(const std::string& path, float minDensity, float maxDensity) const
{
	const auto& densityAxisOpacity = editorOpacity_.getDensityAxis();
	const auto& opacityAxis = editorOpacity_.getOpacityAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();

	assert(densityAxisOpacity.size() == opacityAxis.size());
	assert(densityAxisColor.size() == colorAxis.size());

	nlohmann::json json;
	json["densityAxisOpacity"] = editorOpacity_.getDensityAxis();
	json["opacityAxis"] = editorOpacity_.getOpacityAxis();
	json["densityAxisColor"] = editorColor_.getDensityAxis();
	json["colorAxis"] = editorColor_.getColorAxis();
	json["minDensity"] = minDensity;
	json["maxDensity"] = maxDensity;

	std::ofstream out(path);
	out << json;
	out.close();
}

void TfEditor::loadFromFile(const std::string& path, float& minDensity, float& maxDensity)
{
	nlohmann::json json;
	std::ifstream file(path);
	file >> json;
	file.close();

	std::vector<float> densityAxisOpacity = json["densityAxisOpacity"];
	std::vector<float> opacityAxis = json["opacityAxis"];
	std::vector<float> densityAxisColor = json["densityAxisColor"];
	std::vector<float3> colorAxis = json["colorAxis"];
	minDensity = json["minDensity"];
	maxDensity = json["maxDensity"];

	assert(densityAxisOpacity.size() == opacityAxis.size());
	assert(densityAxisColor.size() == colorAxis.size());

	editorOpacity_.updateControlPoints(densityAxisOpacity, opacityAxis);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

nlohmann::json TfEditor::toJson() const
{
	const auto& densityAxisOpacity = editorOpacity_.getDensityAxis();
	const auto& opacityAxis = editorOpacity_.getOpacityAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();
	//std::vector<nlohmann::json> colorAxis2;
	//std::transform(colorAxis.begin(), colorAxis.end(), std::back_inserter(colorAxis2),
	//	[](float3 color)
	//{
	//	return nlohmann::json::array({ color.x, color.y, color.z });
	//});
	return {
		{"densityAxisOpacity", nlohmann::json(densityAxisOpacity)},
		{"opacityAxis", nlohmann::json(opacityAxis)},
		{"densityAxisColor", nlohmann::json(densityAxisColor)},
		{"colorAxis", nlohmann::json(colorAxis)}
	};
}

void TfEditor::fromJson(const nlohmann::json& s)
{
	const std::vector<float> densityAxisOpacity = s.at("densityAxisOpacity");
	const std::vector<float> opacityAxis = s.at("opacityAxis");
	const std::vector<float> densityAxisColor = s.at("densityAxisColor");
	const std::vector<float3> colorAxis = s.at("colorAxis");
	//std::vector<float3> colorAxis;
	//std::transform(colorAxis2.begin(), colorAxis2.end(), std::back_inserter(colorAxis),
	//	[](nlohmann::json color)
	//{
	//	return make_float3(color[0].get<float>(), color[1].get<float>(), color[2].get<float>());
	//});
	editorOpacity_.updateControlPoints(densityAxisOpacity, opacityAxis);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

bool TfEditor::testIntersectionRectPoint(const ImRect& rect, const ImVec2& point)
{
	return (rect.Min.x <= point.x &&
		rect.Max.x >= point.x &&
		rect.Min.y <= point.y &&
		rect.Max.y >= point.y);
}
