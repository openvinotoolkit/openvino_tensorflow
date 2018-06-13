# Copyright 2018 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.1)

project (ngraph_tensorflow_bridge CXX)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# set(CMAKE_CXX_COMPILER "clang++")
include(ExternalProject)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-comment -Wno-sign-compare")

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Find and build ngraph
ExternalProject_Add(
    ext_ngraph
    GIT_REPOSITORY https://github.com/NervanaSystems/ngraph
    GIT_TAG 894aed9a61d33fa32eb5d2f226fa26733bb416b1
    CMAKE_ARGS -DNGRAPH_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist  -DNGRAPH_USE_PREBUILT_LLVM=${NGRAPH_USE_PREBUILT_LLVM}
    TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/tmp"
    STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/stamp"
    DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/download"
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/build"
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j $(nproc) ngraph
    UPDATE_COMMAND ""
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
)

set( NGRAPH_DIR ${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist)

# Find TensorFlow
find_package(TensorFlow REQUIRED)

# Set the Install Prefix, rpath etc. here (cannot be done in the sub directories)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${TensorFlow_DIR})
endif()
MESSAGE("INSTALL DIR: ${CMAKE_INSTALL_PREFIX}")

add_library(tensorflow_framework_lib SHARED IMPORTED)
set_target_properties( 
    tensorflow_framework_lib 
    PROPERTIES IMPORTED_LOCATION 
    ${TensorFlow_DIR}/libtensorflow_framework.so
)

# Add the directories to be built
add_subdirectory(logging)
add_subdirectory(src)
add_subdirectory(third-party)
add_subdirectory(test)
add_subdirectory(tools)
