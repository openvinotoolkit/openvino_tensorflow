# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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

include_directories(${GTEST_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)

# TODO (avijit)
# Get this location from the CMAKE variables passed during the cmake time
set(TensorFlow_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensorflow")
include_directories(${TensorFlow_SRC_DIR})

# Files that are generated during TF build are here
include_directories(${TensorFlow_SRC_DIR}/bazel-genfiles)

add_library(tensorflow_cc_lib SHARED IMPORTED)
set_target_properties( 
    tensorflow_cc_lib 
    PROPERTIES IMPORTED_LOCATION 
    ${TensorFlow_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so
)

set(SRC
    main.cpp
    graph_exec.cpp
    tf_exec.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(gtest_ngtf ${SRC})
target_link_libraries(
    gtest_ngtf 
    ngraph_device 
    libgtest 
    pthread 
    tensorflow_framework_lib
    tensorflow_cc_lib
)

# The following custom commands are used to create symlinks for various 
# input files needed by the unit test
add_custom_command( 
    TARGET gtest_ngtf
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy.pbtxt 
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy.pbtxt
)
add_custom_command( 
    TARGET gtest_ngtf
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_launchop.pbtxt 
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy_launchop.pbtxt
)

add_subdirectory(python)
