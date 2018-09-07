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

# Get this location from the CMAKE variables passed during the cmake time
if (NOT DEFINED TF_SRC_DIR)
   message(
       FATAL_ERROR 
       "TensorFlow source directory.Use -DTF_SRC_DIR=<tf source directory>"
    )
endif()

set(TensorFlow_SRC_DIR "${TF_SRC_DIR}")
# else()
#     set(TensorFlow_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensorflow")
# endif()

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
    padding.cpp
    conversions.cpp
    graph_rewrites/assign_clusters.cc
    test_utilities.cpp
    test_math_ops.cpp
    test_nn_ops.cpp
    test_array_ops.cpp
    opexecuter.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(gtest_ngtf ${SRC})
message(STATUS "TensorFlow_SRC_DIR: ${TensorFlow_SRC_DIR}")

# The following custom commands are used to create symlinks for various 
# input files needed by the unit test
add_custom_command( 
    TARGET gtest_ngtf
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy.pbtxt 
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy.pbtxt
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_launchop.pbtxt 
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy_launchop.pbtxt
)
add_custom_target( 
    build_tf_cc
    COMMENT "Building TensorFlow C++ library"
    COMMAND 
        ${CMAKE_CURRENT_SOURCE_DIR}/build_tf_cc.sh 
        ${TensorFlow_SRC_DIR} 
        ${NGRAPH_TF_CXX11_ABI}
)

add_dependencies(gtest_ngtf build_tf_cc)

target_link_libraries(
    gtest_ngtf 
    ngraph_device 
    ngraph_lib
    libgtest 
    pthread 
    tensorflow_framework_lib
    tensorflow_cc_lib
)

add_subdirectory(python)
