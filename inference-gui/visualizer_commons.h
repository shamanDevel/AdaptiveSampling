#pragma once

enum ComputationMode
{
	ComputeSuperresolution,
	ComputeAdaptiveSampling,
	ComputeAdaptiveStepsize
};
enum RenderMode
{
	IsosurfaceRendering,
	DirectVolumeRendering
};

extern bool VisualizerEvaluateNetworksInHalfPrecision;
