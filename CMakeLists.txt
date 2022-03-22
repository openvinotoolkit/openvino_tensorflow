# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)

project (openvino_tensorflow CXX)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# set(CMAKE_CXX_COMPILER "clang++")
include(ExternalProject)
include(CMakeDependentOption)
include(cmake/sdl.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wno-comment -Wno-sign-compare -Wno-backslash-newline-escape")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wno-comment -Wno-sign-compare")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14 /wd4308 /wd4146 /wd4703 /wd4244 /wd4819 /EHsc")
endif()

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

if(NOT APPLE)
    # FIXME: Doesn't work for Ubuntu
    execute_process(COMMAND cat /etc/os-release
        OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX MATCH "ID=\"([a-z])+\"" OS_VERSION "${LSB_RELEASE_ID_SHORT}")
    string(REGEX MATCH "\"([a-z])+\"" OS_VERSION "${OS_VERSION}")
    message("OS version is ${OS_VERSION}")
else()
    # Handle the case for MacOS
    # TBD
endif()

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
#       libopenvino_tensorflow.so
#       ...
#     /tensorflow
#       libtensorflow_framework.so.1
#       python/
#           _pywrap....so
# Therefore we are setting two entries in the RPATH:
# 1. $ORIGIN/.
# 2. $ORIGIN/../tensorflow/
#
set(CMAKE_BUILD_RPATH_USE_ORIGIN true)
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/;@loader_path/../tensorflow;")
elseif(DEFINED OPENVINO_TF_RPATH)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow:${OPENVINO_TF_RPATH}")
else()
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow")
endif()

# Find TensorFlow
find_package(TensorFlow REQUIRED)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(OPENVINO_TF_CXX11_ABI "${TensorFlow_CXX_ABI}")
    message( STATUS "OpenVINO™ integration with TensorFlow using CXX11 ABI:  ${OPENVINO_TF_CXX11_ABI}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${OPENVINO_TF_CXX11_ABI}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(OPENVINO_TF_CXX11_ABI "${TensorFlow_CXX_ABI}")
    message( STATUS "OpenVINO™ integration with TensorFlow using CXX11 ABI:  ${OPENVINO_TF_CXX11_ABI}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${OPENVINO_TF_CXX11_ABI}")    
endif()

if(APPLE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(LIBNGRAPH "libngraphd.dylib")
    else()
        set(LIBNGRAPH "libngraph.dylib")
    endif()
elseif(WIN32)
    set(LIBNGRAPH "ngraph.lib")
else()
    set(LIBNGRAPH "libngraph.so")
endif(APPLE)

if(WIN32)
	add_definitions(-DNOMINMAX)
endif()

# Build options
option(UNIT_TEST_ENABLE "Control the building of unit tests" FALSE)
option(UNIT_TEST_TF_CC_DIR "Location where TensorFlow CC library is located" FALSE)
option(USE_OPENVINO_FROM_LOCATION "Use OpenVINO located in OPENVINO_ARTIFACTS_DIR" FALSE)
option(OPENVINO_ARTIFACTS_DIR "Where would OpenVINO be installed after build" FALSE)

set(InferenceEngine_DIR ${OPENVINO_ARTIFACTS_DIR}/runtime/cmake)
find_package(InferenceEngine REQUIRED)

set(ngraph_DIR ${OPENVINO_ARTIFACTS_DIR}/runtime/cmake)
find_package(ngraph QUIET)

include_directories(${InferenceEngine_INCLUDE_DIRS})
if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    set(TBB_DIR ${OPENVINO_ARTIFACTS_DIR}/runtime/3rdparty/tbb/cmake)
    find_package(TBB COMPONENTS tbb tbbmalloc)
endif()

message(STATUS "UNIT_TEST_TF_CC_DIR:        ${TF_PRE_BUILT_LOCATION}")
if(UNIT_TEST_TF_CC_DIR)
    # Check if the path specified is ABSOLUTE or RELATVE
    if (NOT IS_ABSOLUTE ${UNIT_TEST_TF_CC_DIR})
        set(UNIT_TEST_TF_CC_DIR ${CMAKE_CURRENT_BINARY_DIR}/${UNIT_TEST_TF_CC_DIR})
    endif()

    # Create absolute path for the directory
    get_filename_component(
        TF_PRE_BUILT_LOCATION
        "${UNIT_TEST_TF_CC_DIR}"
        ABSOLUTE
    )
    if (NOT EXISTS ${TF_PRE_BUILT_LOCATION})
        message(FATAL_ERROR
            "TensorFlow pre-built directory doesn't exist: " ${TF_PRE_BUILT_LOCATION} )
    endif()
endif()

message(STATUS "UNIT_TEST_ENABLE:           ${UNIT_TEST_ENABLE}")
message(STATUS "OPENVINO_ARTIFACTS_DIR:     ${OPENVINO_ARTIFACTS_DIR}")
message(STATUS "USE_PRE_BUILT_OPENVINO:       ${USE_PRE_BUILT_OPENVINO}")
message(STATUS "OPENVINO_VERSION: ${OPENVINO_VERSION}")

if (${OPENVINO_VERSION} MATCHES "master")
  # master works with checks of latest release
  add_definitions(-DOPENVINO_2022_1=1)
elseif (${OPENVINO_VERSION} MATCHES "2022.1.0")
  add_definitions(-DOPENVINO_2022_1=1)
else()
  message(FATAL_ERROR "Unsupported OpenVINO version: ${OPENVINO_VERSION")
endif()

if(WIN32)
    add_definitions(-DBUILD_API=1)    
endif()

if(OS_VERSION STREQUAL "\"centos\"")
    set(LIB "lib64")
else()
    set(LIB "lib/intel64")
endif()

if(ngraph_FOUND)
    set(NGRAPH_INSTALL_DIR ${OPENVINO_ARTIFACTS_DIR}/runtime)
else()
    set(NGRAPH_INSTALL_DIR ${OPENVINO_ARTIFACTS_DIR})
endif()

add_library(tensorflow_frontend_lib SHARED IMPORTED)                                                                       
set_target_properties(
    tensorflow_frontend_lib
    PROPERTIES IMPORTED_LOCATION
    ${NGRAPH_INSTALL_DIR}/${LIB}/libopenvino_tensorflow_fe.so
)

add_library(openvino_lib SHARED IMPORTED)                                                                       
set_target_properties(
    openvino_lib
    PROPERTIES IMPORTED_LOCATION
    ${NGRAPH_INSTALL_DIR}/${LIB}/libopenvino.so
)

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")

# Add the directories to be built
add_subdirectory(third-party)
add_subdirectory(logging)
add_subdirectory(openvino_tensorflow)
add_subdirectory(tools)
add_subdirectory(ocm/OCM)

# The following targets depend on the Tensorflow source code directory
# Get the absolute file name for the source
get_filename_component(
    TensorFlow_SRC_DIR
    "${TF_SRC_DIR}"
    ABSOLUTE
)

add_subdirectory(examples)
if (DEFINED TF_SRC_DIR)
    message(STATUS "TensorFlow_SRC_DIR: ${TensorFlow_SRC_DIR}")
    add_subdirectory(examples/classification_sample)
else()
    message(
        STATUS
        "TensorFlow source directory not provided. "
        "C++ Examples won't be built"
    )
endif()

if (UNIT_TEST_ENABLE)
    if (NOT DEFINED TF_SRC_DIR)
        message(
            STATUS
            "TensorFlow source directory not provided. "
            "C++ unit tests won't be built"
        )
    else()
        if (NOT EXISTS ${TensorFlow_SRC_DIR})
            message(
                STATUS
                "TensorFlow source directory doesn't exist"
            )
        endif()

        # Check if the path specified is ABSOLUTE or RELATVE
        if (NOT IS_ABSOLUTE ${TF_SRC_DIR})
            set(TF_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TF_SRC_DIR})
        endif()
    endif()

    add_subdirectory(test)
    message(STATUS "unit tests enabled")
endif()

