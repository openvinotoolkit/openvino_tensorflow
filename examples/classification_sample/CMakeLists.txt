# ******************************************************************************
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

# Get this location from the CMAKE variables passed during the cmake time
if (NOT DEFINED TF_SRC_DIR)
   message(
       FATAL_ERROR
       "No TensorFlow source directory.Use -DTF_SRC_DIR=<tf source directory>"
    )
endif()

message(STATUS "TF Src Dir: ${TensorFlow_SRC_DIR}")
include_directories(${TensorFlow_SRC_DIR})

# Files that are generated during TF build are here
include_directories(${TensorFlow_SRC_DIR}/bazel-genfiles)
include_directories(${TensorFlow_SRC_DIR}/bazel-bin)

add_library(tensorflow_cc_lib SHARED IMPORTED)
message(${TensorFlow_VERSION})
if(WIN32)
    set_target_properties(
    tensorflow_cc_lib
    PROPERTIES IMPORTED_LOCATION
    ${TensorFlow_SRC_DIR}/bazel-bin/tensorflow/tensorflow_cc.dll.if.lib)
else()
  if(NOT(TensorFlow_VERSION LESS 2.0))
      set_target_properties(
          tensorflow_cc_lib
          PROPERTIES IMPORTED_LOCATION
          ${TensorFlow_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2)
  else()
      set_target_properties(
      tensorflow_cc_lib
      PROPERTIES IMPORTED_LOCATION
      ${TensorFlow_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.1)
  endif()
endif()

set(SRC
    thread_safe_queue.h
    utils.cc
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "TensorFlow_SRC_DIR: ${TensorFlow_SRC_DIR}")

set(APP_NAME infer_image)
add_executable(
    ${APP_NAME} ${SRC} main.cc
)

if (APPLE)
    set(OPENVINO_TF_CXX11_ABI 0)
    add_definitions(-DTEST_SINGLE_INSTANCE)
endif()

if(WIN32)
	get_target_property(tensorflow_cc_lib_value tensorflow_cc_lib IMPORTED_LOCATION)
	target_link_libraries(
		${APP_NAME}
		openvino_tensorflow
		${TensorFlow_FRAMEWORK_LIBRARY}
		${tensorflow_cc_lib_value}
		absl_synchronization
		${InferenceEngine_LIBRARIES} ${TBB_IMPORTED_TARGETS}
	)
else()
  target_link_libraries(
      ${APP_NAME}
      openvino_tensorflow
      pthread
      ${TensorFlow_FRAMEWORK_LIBRARY}
      tensorflow_cc_lib
      absl_synchronization
      ${InferenceEngine_LIBRARIES} ${TBB_IMPORTED_TARGETS}
  )
endif()

if (DEFINED OPENVINO_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${OPENVINO_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()

install(TARGETS ${APP_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/classification_sample)

