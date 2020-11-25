# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)

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
set_target_properties(
    tensorflow_cc_lib
    PROPERTIES IMPORTED_LOCATION
    ${TensorFlow_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2
)

set(SRC
    inference_engine.h
    thread_safe_queue.h
    inference_engine.cc
    tf_label_image_utils.cc
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "TensorFlow_SRC_DIR: ${TensorFlow_SRC_DIR}")

set(APP_NAME infer_single_network)
add_executable(
    ${APP_NAME} ${SRC} infer_single_network.cc
)

if (APPLE)
    set(NGRAPH_TF_CXX11_ABI 0)
    add_definitions(-DTEST_SINGLE_INSTANCE)
endif()

target_link_libraries(
    ${APP_NAME}
    ngraph_bridge
    ngraph_lib
    pthread
    ${TensorFlow_FRAMEWORK_LIBRARY}
    tensorflow_cc_lib
    absl_synchronization
    ${InferenceEngine_LIBRARIES} ${TBB_IMPORTED_TARGETS}
)

set(APP_NAME_MULTI infer_multiple_networks)
add_executable(
    ${APP_NAME_MULTI} ${SRC} infer_multiple_networks.cc
)

target_link_libraries(
    ${APP_NAME_MULTI}
    ngraph_bridge
    ngraph_lib
    pthread
    ${TensorFlow_FRAMEWORK_LIBRARY}
    tensorflow_cc_lib
    absl_synchronization
    ${InferenceEngine_LIBRARIES} ${TBB_IMPORTED_TARGETS}
)

if (DEFINED NGRAPH_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()

install(TARGETS ${APP_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/cpp/inference)
install(TARGETS ${APP_NAME_MULTI} DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/cpp/inference)
