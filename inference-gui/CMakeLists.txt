cmake_minimum_required(VERSION 3.17)

OPTION(BUILD_INFERENCE_GUI "Build inference gui project" ON)

if (${BUILD_INFERENCE_GUI})
	# your configuration
	set(INFERENCE_FILES
		main.cpp
		utils.h
		visualizer_commons.h
		visualizer.h
		visualizer.cpp
		tf_editor.h
		tf_editor.cpp
		visualizer_kernels.h
		visualizer_kernels.cu
		quad_drawer.h
		quad_drawer.cpp
		camera_gui.h
		camera_gui.cpp
		superres_model.h
		superres_model.cpp
		background_worker.h
		adaptive_model.h
		adaptive_model.cpp
		importance_sampling_model.h
		importance_sampling_model.cpp
		hsv_normalization.h
		hsv_normalization.cpp
		stepsize_model.h
		stepsize_model.cpp

		imgui/imconfig.h
		imgui/imgui.h
		imgui/imgui.cpp
		imgui/imgui_demo.cpp
		imgui/imgui_draw.cpp
		imgui/imgui_internal.h
		imgui/imgui_widgets.cpp
		imgui/imstb_rectpack.h
		imgui/imstb_textedit.h
		imgui/imstb_truetype.h
		imgui/imgui_impl_opengl3.cpp
		imgui/imgui_impl_opengl3.h
		imgui/imgui_impl_glfw.cpp
		imgui/imgui_impl_glfw.h
		imgui/IconsFontAwesome5.h
		imgui/imgui_extension.h
		imgui/imgui_extension.cpp

		tinyformat.h
		../third-party/nlohmann/json.hpp
		../third-party/lodepng/lodepng.cpp
		../third-party/lodepng/lodepng.h
		../third-party/portable-file-dialogs/portable-file-dialogs.h
		)
	set(INFERENCE_NAME InferenceGui CACHE INTERNAL "target name for the inference")

	# create the executable
	cuda_add_executable(${INFERENCE_NAME}
		${INFERENCE_FILES}
		)
	target_include_directories(${INFERENCE_NAME}
		PRIVATE 
		${TORCH_INCLUDE_DIR} 
		${LIBRARY_INCLUDE_DIR} 
		${GLM_INCLUDE_DIRS} 
		${GLEW_INCLUDE_DIR} 
		${GLFW_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/third-party/cuMat
		${CMAKE_SOURCE_DIR}/third-party/nlohmann
		${CMAKE_SOURCE_DIR}/third-party/lodepng
		${CMAKE_SOURCE_DIR}/third-party/thread-pool/include
		${CMAKE_SOURCE_DIR}/third-party/portable-file-dialogs
		${OpenCV_INCLUDE_DIRS}
		)
	target_link_libraries(${INFERENCE_NAME}
		${LIBRARY_NAME} 
		${OPENGL_LIBRARY} 
		${GLEW_LIBRARY_RELEASE}
		${GLEW_SHARED_LIBRARY_RELEASE}
		${GLFW_LIBRARIES}
		${OpenCV_LIBRARIES})
	set_property(TARGET ${INFERENCE_NAME} PROPERTY CXX_STANDARD 17)
	add_custom_command(TARGET ${INFERENCE_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${INFERENCE_NAME}> ${CMAKE_SOURCE_DIR}/bin/${INFERENCE_NAME}${CMAKE_EXECUTABLE_SUFFIX}


		COMMENT "Copies the executable to bin/"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
		
		VERBATIM
		)
	target_compile_definitions(${INFERENCE_NAME} 
		PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLEW=1)
	set_property(TARGET ${INFERENCE_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

	# Resources
	file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
					 "${CMAKE_BINARY_DIR}/CMakeRC.cmake")
	include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
	cmrc_add_resource_library(
		${INFERENCE_NAME}-resources

		ALIAS ${INFERENCE_NAME}::rc
		NAMESPACE ui

		resources/fa-regular-400.ttf
		resources/fa-solid-900.ttf
		resources/hsv2rgb.pt
		resources/rgb2hsv.pt
		)
	target_link_libraries(${INFERENCE_NAME} ${INFERENCE_NAME}::rc)


endif()