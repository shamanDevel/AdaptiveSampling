cmake_minimum_required(VERSION 3.10)
project(IsoSurfaceSuperResolution)

####################################
# C++ standard
####################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

####################################
# GENERAL THIRD-PARTY DEPENDENCIES
####################################

# CUDA is always required
find_package(CUDA REQUIRED)
if(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
	if (WIN32) # inference-gui
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
	else() # server
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 6.1)
	endif()
	LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
	message(STATUS "cuda flags: ${ARCH_FLAGS}")
endif()
list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
set(MY_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
#LIST(APPEND CUDA_NVCC_FLAGS "--keep") #for debugging the .ptx files
enable_language(CUDA)

# PYTHON
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
get_filename_component(PYTHON_DIRECTORY ${PYTHON_EXECUTABLE} DIRECTORY)

####################################
# PYTORCH
####################################
# find installation path
if(NOT DEFINED ${TORCH_PATH})
	# query torch path from python
	execute_process(COMMAND python -c "import sys; import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0], file=sys.stderr)" ERROR_VARIABLE TORCH_FIRST_INCLUDE_DIR)
	get_filename_component(TORCH_ROOT ${TORCH_FIRST_INCLUDE_DIR}/../ ABSOLUTE)
	set(TORCH_PATH "${TORCH_ROOT}" CACHE FILEPATH "path to pytorch in the python installation")
	if(NOT (EXISTS ${TORCH_PATH}))
		message( FATAL_ERROR "Pytorch not found, is it not installed in the python distribution ${PYTHON_DIRECTORY}?")
	else()
		message(STATUS "Torch found at ${TORCH_PATH}")
	endif()
endif(NOT DEFINED ${TORCH_PATH})
# ask Torch's CMake configuration
set(TORCH_CONFIG_PATH "${TORCH_PATH}/share/cmake/Torch" CACHE FILEPATH "possible path where TorchConfig.cmake is located")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CONFIG_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Torch Cxx flags: ${TORCH_CXX_FLAGS}")
# overwrite nvcc flags (because of wrong architecture in Torch)
set(CUDA_NVCC_FLAGS ${MY_CUDA_NVCC_FLAGS})
message(STATUS "overwrite cuda flags: ${CUDA_NVCC_FLAGS}")
# get libraries (hard coded)
set(TORCH_LIBRARY_NAMES
	c10 c10_cuda caffe2_nvrtc torch_python _C)
set(TORCH_LIBRARIES ${TORCH_LIBRARY})
FOREACH(LIB_NAME ${TORCH_LIBRARY_NAMES})
  set(LIB_VAR "TORCH_LIB_${LIB_NAME}") # Name of the variable which stores result of the search
  FIND_LIBRARY(${LIB_VAR} ${LIB_NAME} PATHS ${TORCH_PATH}/lib)
  if(${LIB_VAR})
    list(APPEND TORCH_LIBRARIES ${${LIB_VAR}})
  endif()
ENDFOREACH()
message(STATUS "Torch: full library list: ${TORCH_LIBRARIES}")
# copy shared library to bin/
file(MAKE_DIRECTORY bin)
file(GLOB TORCH_SHARED_LIBRARIES
	${TORCH_PATH}/lib/*${CMAKE_SHARED_LIBRARY_SUFFIX})
message(STATUS "Torch: shared libraries to copy: ${TORCH_SHARED_LIBRARIES}")
file(COPY ${TORCH_SHARED_LIBRARIES} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
# get include directories
set(TORCH_INCLUDE_DIR "${TORCH_PATH}/include;${TORCH_PATH}/include/torch/csrc/api/include" CACHE FILEPATH "include directory for the pytorch headers")
message(STATUS "Torch: include directories: ${TORCH_INCLUDE_DIR}")

####################################
# OpenGL
####################################
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW)
find_package(GLM REQUIRED)
# if glfw failed, check using PkgConfig
if(NOT GLFW_FOUND)
	message(STATUS "GLFW could not be found with normal lookup, use PkgConfig instead")
	find_package(PkgConfig REQUIRED)
	pkg_search_module(GLFW REQUIRED glfw3)
else()
	set(GLFW_LIBRARIES ${GLFW_LIBRARY})
	set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
endif()
# copy shared libraries
if (WIN32)
	# glew dll if running on windows
	string(REPLACE "/lib/" "/bin/" GLEW_BINARY_RELEASEa "${GLEW_LIBRARY_RELEASE}")
	string(REPLACE "/lib/" "/bin/" GLEW_BINARY_RELEASEa "${GLEW_SHARED_LIBRARY_RELEASE}")
	string(REPLACE ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX} GLEW_BINARY_RELEASE ${GLEW_BINARY_RELEASEa})
	file(COPY ${GLEW_BINARY_RELEASE} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
else()
	# copy glew, glfw, glm
	file(COPY ${GLEW_SHARED_LIBRARY_RELEASE} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
endif()

####################################
# THE LIBRARY
####################################
add_subdirectory(renderer)

####################################
# PYTHON APPLICATION
####################################
add_subdirectory(network)

####################################
# TEST APPLICATION
# depend on the library
####################################
add_subdirectory(inference-gui)