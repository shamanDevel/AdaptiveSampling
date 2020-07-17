cmake_minimum_required(VERSION 3.10)

# OUTPUT CACHE VARIABLES:
# BINDING_NAME
# LIBRARY_INCLUDE_DIR
# LIBRARY_LINK_LIBRARIES

OPTION(RENDERER_BUILD_RENDERER "Build raytracer (volume, camera, settings, renderer, tf)" ON)
OPTION(RENDERER_BUILD_MESH_DRAWER "Build mesh drawer (requires OpenGL)" ON)
OPTION(RENDERER_BUILD_SAMPLING "Build sampling (sampling.h + halton)" ON)
OPTION(RENDERER_BUILD_INPAINTING "Build inpainting" ON)

# your configuraton
set(LIBRARY_HEADER_FILES
	lib.h
	volume.h
	renderer.h
	settings.h
	camera.h
	commons.h
	helper_math.h
	sampling.h
	mesh_drawer.h
	halton_sampler.h
	inpainting.h
	tf_texture_1d.h
	regular_interp1d.h
	)
set(LIBRARY_CUDA_FILES
	sampling.cu
	mesh_drawer.cu
	renderer.cu
	inpainting_fast.cu
	inpainting_pde.cu
	tf_texture_1d.cu
	volume.cu
	regular_interp1d.cu
	)
set(LIBRARY_SOURCE_FILES
	volume.cpp
	settings.cpp
	camera.cpp
	tf_texture_1d.cpp
	jit.cpp # bindings, also needed in the test application
	)
# the target name of the library
set(LIBRARY_NAME Renderer CACHE INTERNAL "the target name of the library, also used for the binding")

#add_library(${LIBRARY_NAME}_cuda OBJECT ${LIBRARY_CUDA_FILES})

# the library, compiled as a shared library
cuda_add_library(${LIBRARY_NAME} SHARED
	${LIBRARY_HEADER_FILES} ${LIBRARY_SOURCE_FILES} ${LIBRARY_CUDA_FILES})
set_target_properties(${LIBRARY_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 11)
# get include path to the first library header, to be used in the test application
list(GET ${LIBRARY_HEADER_FILES} 0 LIBRARY_INCLUDE_DIRb)
get_filename_component(LIBRARY_INCLUDE_DIRa ${LIBRARY_INCLUDE_DIRb} ABSOLUTE)
get_filename_component(LIBRARY_INCLUDE_DIRc ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
set(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRc} CACHE FILEPATH "include directory of the custom library") 
target_include_directories(${LIBRARY_NAME}
	PRIVATE 
		${PYTHON_INCLUDE_DIR} 
		${CMAKE_SOURCE_DIR}/third-party/cuMat 
		${GLM_INCLUDE_DIRS}
		${GLEW_INCLUDE_DIR} 
		${GLFW_INCLUDE_DIRS}
	PUBLIC ${TORCH_INCLUDE_DIR}
	INTERFACE ${LIBRARY_INCLUDE_DIR}
	)
# libraries, also used in the test application
set(LIBRARY_LINK_LIBRARIES
	${PYTHON_LIBRARY}
	${TORCH_LIBRARIES})
target_link_libraries(${LIBRARY_NAME}
	${LIBRARY_LINK_LIBRARIES}
	${OPENGL_LIBRARY} 
	${GLEW_LIBRARY_RELEASE} 
	${GLEW_SHARED_LIBRARY_RELEASE} 
	${GLFW_LIBRARIES})
target_compile_definitions(${LIBRARY_NAME}
	PRIVATE BUILD_MAIN_LIB)
set_property(TARGET ${LIBRARY_NAME} PROPERTY CXX_STANDARD 17)
# filter what to compile
# this is needed because the trainings server does not support the openGL stuff
if(${RENDERER_BUILD_RENDERER})
	target_compile_definitions(${LIBRARY_NAME} PUBLIC RENDERER_HAS_RENDERER)
endif()
if(${RENDERER_BUILD_MESH_DRAWER})
	target_compile_definitions(${LIBRARY_NAME} PUBLIC RENDERER_HAS_MESH_DRAWER)
endif()
if(${RENDERER_BUILD_SAMPLING})
	target_compile_definitions(${LIBRARY_NAME} PUBLIC RENDERER_HAS_SAMPLING)
endif()
if(${RENDERER_BUILD_INPAINTING})
	target_compile_definitions(${LIBRARY_NAME} PUBLIC RENDERER_HAS_INPAINTING)
endif()
# post-build commands, copies the dll to the bin/ folder
get_filename_component(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
add_custom_command(TARGET ${LIBRARY_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LIBRARY_NAME}> ${CMAKE_SOURCE_DIR}/bin/${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}

	COMMENT "Copies the libarary .dll to bin/"
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
	
	VERBATIM
	)

# debugging: keep .ptx files of the kernels
#set_property(TARGET ${LIBRARY_NAME}_cuda PROPERTY CUDA_PTX_COMPILATION ON)
#target_compile_options(${LIBRARY_NAME} PRIVATE "--keep")