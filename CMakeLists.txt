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

# In order to compile ngraph-tf with memory leak detection, run `cmake` with `-DCMAKE_BUILD_TYPE=Sanitize`.
# N.B.: This *will* crash python unit tests because ngraph-tf will be loaded "too late" via `dlopen`,
# so only use this with C++ tests.
# (In theory using `LD_PRELOAD` should address the python issue, but it doesn't appear to work on OS X, at least.)
# If there are any memory leaks, then upon running the binary a report will be automatically generated.
SET(CMAKE_CXX_FLAGS_SANITIZE
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -D_LIBCPP_HAS_NO_ASAN -fsanitize-address-use-after-scope"
    CACHE STRING "Flags used by the C++ compiler during sanitized builds."
    FORCE )
SET(CMAKE_C_FLAGS_SANITIZE
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -D_LIBCPP_HAS_NO_ASAN -fsanitize-address-use-after-scope"
    CACHE STRING "Flags used by the C compiler during sanitized builds."
    FORCE )
SET(CMAKE_EXE_LINKER_FLAGS_SANITIZE
    "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -D_LIBCPP_HAS_NO_ASAN -fsanitize-address-use-after-scope"
    CACHE STRING "Flags used for linking binaries during sanitized builds."
    FORCE )
SET(CMAKE_SHARED_LINKER_FLAGS_SANITIZE
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address -D_LIBCPP_HAS_NO_ASAN -fsanitize-address-use-after-scope"
    CACHE STRING "Flags used by the shared libraries linker during coverage builds."
    FORCE )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_SANITIZE
    CMAKE_C_FLAGS_SANITIZE
    CMAKE_EXE_LINKER_FLAGS_SANITIZE
    CMAKE_SHARED_LINKER_FLAGS_SANITIZE)

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Invoke a command to determine how many CPU cores we have, and set
# NUM_MAKE_PROCESSES accordingly so we know what number to pass to make -j.
if(APPLE)
   set (PROCESSOR_COUNT_COMMAND sysctl -n hw.physicalcpu)
else()
   set (PROCESSOR_COUNT_COMMAND nproc)
endif()

execute_process(
    COMMAND ${PROCESSOR_COUNT_COMMAND}
    RESULT_VARIABLE NPROC_RESULT
    OUTPUT_VARIABLE NUM_MAKE_PROCESSES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Default to four jobs if the command fails.
if(NPROC_RESULT)
    message (WARNING "Unable to detect number of processors. Building nGraph with make -j4.")
    set(NUM_MAKE_PROCESSES 4)
endif()

# Need to setup the RPATH here - else it won't work.
# During installation, a Python pip package is created which when
# installed is located in the same level as the tensorflow directory
# site-packages/
#     /ngraph
#       libngraph_bridge.so
#       ...
#     /tensorflow
#       libtensorflow_framework.so
#       python/
#           _pywrap....so
# Therefore we are setting two entries in the RPATH:
# 1. $ORIGIN/.
# 2. $ORIGIN/../tensorflow/
# 
set(CMAKE_MACOSX_RPATH 1)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/;@loader_path/../tensorflow;")
else()
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow")
endif()

# Find TensorFlow
find_package(TensorFlow REQUIRED)

add_library(tensorflow_framework_lib SHARED IMPORTED)
set_target_properties( 
    tensorflow_framework_lib 
    PROPERTIES IMPORTED_LOCATION 
    ${TensorFlow_DIR}/libtensorflow_framework.so
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(NGRAPH_TF_CXX11_ABI "${TensorFlow_CXX_ABI}")
    message( STATUS "nGraph-TensorFlow using CXX11 ABI:  ${NGRAPH_TF_CXX11_ABI}" )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_TF_CXX11_ABI}")

if(APPLE)
    set(LIBNGRAPH_SO "libngraph.dylib")
else()
    set(LIBNGRAPH_SO "libngraph.so")
endif(APPLE)

# Find and build ngraph
ExternalProject_Add(
    ext_ngraph
    GIT_REPOSITORY https://github.com/NervanaSystems/ngraph
    GIT_TAG v0.8.0
    CMAKE_ARGS 
        -DNGRAPH_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist 
        -DNGRAPH_USE_CXX_ABI=${TensorFlow_CXX_ABI}
        -DNGRAPH_UNIT_TEST_ENABLE=FALSE
        -DNGRAPH_TOOLS_ENABLE=FALSE 
        -DNGRAPH_DEX_ONLY=TRUE         
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/tmp"
    STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/stamp"
    DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/download"
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/build"
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j ${NUM_MAKE_PROCESSES} ngraph
    UPDATE_COMMAND ""
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
)

set( NGRAPH_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist)


add_library(ngraph_lib SHARED IMPORTED)
set_target_properties( 
    ngraph_lib 
    PROPERTIES IMPORTED_LOCATION 
    ${NGRAPH_INSTALL_DIR}/lib/${LIBNGRAPH_SO}
)
add_dependencies(ngraph_lib ext_ngraph)

option(UNIT_TEST_ENABLE "Control the building of unit tests" FALSE)

# Add the directories to be built
add_subdirectory(logging)
add_subdirectory(src)
add_subdirectory(third-party)
if (UNIT_TEST_ENABLE)
    if (NOT DEFINED TF_SRC_DIR)
        message(FATAL_ERROR "Provide TensorFlow source directory: -DTF_SRC_DIR=<directory>")
    endif()
    add_subdirectory(test)
    message(STATUS "unit tests enabled")
endif()
