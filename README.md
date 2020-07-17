# Learning adaptive sampling and reconstruction for Volume Visualization

Currently supported:
 - Iso-surface rendering with screen-space shading
 - Direct Volume Rendering
 - Temporal consistency and reprojection
 - adaptive sampling with an importance map in screen space
 - adaptive sampling in object space by changing the step size

## Project structure:
 - renderer: a shared library exposing PyTorch operation that contains the rendering core (C++, CUDA)
 - network: super-resolution network training and testing code (Python, PyTorch)
 - inference-gui: interactive gui combining the renderer and networks, allows to test all available options (C++, OpenGL)

See the release page for binaries, datasets and pretrained networks
 
## Requirements

 - CUDA >= 1.1
 - Python >= 3.6
 - PyTorch >= 1.5
 - OpenGL
 
Tested with CUDA 10.1, Python 3.6, PyTorch 1.5, Windows 10 and Ubuntu 18