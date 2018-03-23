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

project (ngraph_plugin CXX)
set(NGRAPH_VERSION 0.0)
set(NGRAPH_TENSORFLOW_BRIDGE_LIB_NAME ngraph_plugin)

# set(CMAKE_CXX_COMPILER "clang++")
include(ExternalProject)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-comment -Wno-sign-compare")

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

execute_process(
    COMMAND 
    python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TEMP
)
string(REGEX REPLACE "\n$" "" TENSORFLOW_INCLUDE_DIR "${TEMP}")

execute_process(
    COMMAND 
    python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE TEMP
)
string(REGEX REPLACE "\n$" "" TENSORFLOW_DIR "${TEMP}")


# Link the ngraph stuff
ExternalProject_Add(
    ext_ngraph
    GIT_REPOSITORY https://github.com/NervanaSystems/ngraph
    GIT_TAG v0.1.0-rc1
    CMAKE_ARGS -DNGRAPH_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist -DNGRAPH_USE_PREBUILT_LLVM=TRUE
    TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/tmp"                           
    STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/stamp"                       
    DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/download"                 
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/build"                      
    UPDATE_COMMAND ""
)


set( NGRAPH_DIR ${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist)

MESSAGE( STATUS "TENSORFLOW_DIR: " ${TENSORFLOW_DIR} )
MESSAGE( STATUS "TENSORFLOW_INCLUDE_DIR: " ${TENSORFLOW_INCLUDE_DIR} )
MESSAGE( STATUS "NGRAPH_DIR: " ${NGRAPH_DIR} )

include_directories( ${TENSORFLOW_INCLUDE_DIR} )
include_directories( ${TENSORFLOW_INCLUDE_DIR}/external/nsync/public )
include_directories( ${NGRAPH_DIR}/include )

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set( NGRAPH_INSTALL_DIR ${TENSORFLOW_DIR}/plugins)
set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_DIR})

set(CMAKE_INSTALL_RPATH "${NGRAPH_DIR}/lib")

# Add the directories to be built
add_subdirectory(src)

