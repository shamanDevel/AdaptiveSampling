cmake_minimum_required(VERSION 3.10)

# configuration
OPTION(TRAIN_USE_PYTHON_PROJECT "Generate Python project for the training (VS)" ON)

set(TRAIN_PROJECT_FILE "Network.pyproj" CACHE STRING "The filename of the pyproj project file")

if (${TRAIN_USE_PYTHON_PROJECT})
	set(TRAIN_FULL_PROJECT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${TRAIN_PROJECT_FILE}")
	if(EXISTS ${TRAIN_FULL_PROJECT_FILE})
		message(STATUS "Python project file ${TRAIN_FULL_PROJECT_FILE} already exists, don't recreate it")
	else()
		message(STATUS "Python project file ${TRAIN_FULL_PROJECT_FILE} does not exist, recreate it")
		# set parameters for the project
		set(TRAIN_PROJECT_HOME "${CMAKE_CURRENT_SOURCE_DIR}")
		set(TRAIN_STARTUP_FILE "")
		set(TRAIN_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
		# TODO: files
		# write project file
		configure_file(${CMAKE_SOURCE_DIR}/cmake/python_project.pyproj ${TRAIN_FULL_PROJECT_FILE})
	endif()
	# add custom project to solution
	get_filename_component(TRAIN_PROJECT_NAME ${TRAIN_FULL_PROJECT_FILE} NAME_WE)
	INCLUDE_EXTERNAL_MSPROJECT(${TRAIN_PROJECT_NAME} ${TRAIN_FULL_PROJECT_FILE} ${LIBRARY_NAME})
endif()