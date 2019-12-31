# Copyright 2018-2019 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.4)

project (ngraph_tensorflow_bridge CXX)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# set(CMAKE_CXX_COMPILER "clang++")
include(ExternalProject)
include(CMakeDependentOption)
include(cmake/sdl.cmake)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-comment -Wno-sign-compare -Wno-backslash-newline-escape")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-comment -Wno-sign-compare")
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
#       libngraph_bridge.so
#       ...
#     /tensorflow
#       libtensorflow_framework.so.1
#       python/
#           _pywrap....so
# Therefore we are setting two entries in the RPATH:
# 1. $ORIGIN/.
# 2. $ORIGIN/../tensorflow/
#
set(CMAKE_MACOSX_RPATH 1)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/;@loader_path/../tensorflow;")
elseif(DEFINED NGRAPH_TF_RPATH)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow:${NGRAPH_TF_RPATH}")
else()
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../tensorflow")
endif()

# Find TensorFlow
find_package(TensorFlow REQUIRED)



if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(NGRAPH_TF_CXX11_ABI "${TensorFlow_CXX_ABI}")
    message( STATUS "nGraph-TensorFlow using CXX11 ABI:  ${NGRAPH_TF_CXX11_ABI}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_TF_CXX11_ABI}")
endif()

if(NGRAPH_BRIDGE_STATIC_LIB_ENABLE)
    set(LIBNGRAPH "libngraph.a")
    set(LIBCPU_BACKEND_A "libcpu_backend.a")
    set(LIBINTERPRETER_BACKEND_A "libinterpreter_backend.a")
else()
    set(LIBNGRAPH "libngraph.so")
endif()

# TODO[Sindhu]
# Should make static library changes for APPLE 
# after Ngraph makes required changes for Mac build

if(APPLE)
    set(LIBNGRAPH "libngraph.dylib")
endif(APPLE)

# Build options
option(USE_PRE_BUILT_NGRAPH "Use pre-built nGraph located in NGRAPH_ARTIFACTS_DIR" FALSE)
option(NGRAPH_ARTIFACTS_DIR "Where would nGraph be installed after build" FALSE)
option(UNIT_TEST_ENABLE "Control the building of unit tests" FALSE)
option(UNIT_TEST_TF_CC_DIR "Location where TensorFlow CC library is located" FALSE)
option(NGRAPH_DISTRIBUTED_ENABLE "Add distributed mode to the CPU backend" FALSE)
option(NGRAPH_BRIDGE_STATIC_LIB_ENABLE "Enable build nGraph bridge static library" FALSE)

find_package(PlaidML CONFIG)
option(NGRAPH_PLAIDML_ENABLE "Build PlaidML backend" ${PLAIDML_FOUND})

# Validate the options
if (NGRAPH_PLAIDML_ENABLE)
    if (NOT PLAIDML_FOUND)
        message(FATAL_ERROR "Unable to find PlaidML; use \"pip install plaidml\" to install it.")
    endif()
endif()

if(USE_PRE_BUILT_NGRAPH)
    if(NOT NGRAPH_ARTIFACTS_DIR)
        message(
            FATAL_ERROR
            "USE_PRE_BUILT_NGRAPH is ON but NGRAPH_ARTIFACTS_DIR is missing"
        )
    endif()
    # Check if the path specified is ABSOLUTE or RELATVE
    if (NOT IS_ABSOLUTE ${NGRAPH_ARTIFACTS_DIR})
        set(NGRAPH_ARTIFACTS_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NGRAPH_ARTIFACTS_DIR})
    endif()

    # Create absolute path for the directory
    get_filename_component(
        NGRAPH_ARTIFACTS_DIR
        "${NGRAPH_ARTIFACTS_DIR}"
        ABSOLUTE
    )
    if (NOT EXISTS ${NGRAPH_ARTIFACTS_DIR})
        message(FATAL_ERROR
            "nGraph artifacts directory doesn't exist: " ${NGRAPH_ARTIFACTS_DIR} )
    endif()
else()
    set(NGRAPH_ARTIFACTS_DIR ${CMAKE_CURRENT_BINARY_DIR}/ngraph/ngraph_dist)
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

# Enable build target CPU features
if (NOT DEFINED NGRAPH_TARGET_ARCH)
    set(
        NGRAPH_TARGET_ARCH
        native
        CACHE STRING "Target CPU architecture to build for.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${NGRAPH_TARGET_ARCH}")

if (NOT DEFINED NGRAPH_TUNE_ARCH)
    set(
        NGRAPH_TUNE_ARCH
        native
        CACHE STRING "Target CPU architecture to build for.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=${NGRAPH_TARGET_ARCH}")

message(STATUS "UNIT_TEST_ENABLE:           ${UNIT_TEST_ENABLE}")

message(STATUS "NGRAPH_ARTIFACTS_DIR:       ${NGRAPH_ARTIFACTS_DIR}")
message(STATUS "USE_PRE_BUILT_NGRAPH:       ${USE_PRE_BUILT_NGRAPH}")
message(STATUS "NGRAPH_DISTRIBUTED_ENABLE:  ${NGRAPH_DISTRIBUTED_ENABLE}")
message(STATUS "NGRAPH_PLAIDML_ENABLE:      ${NGRAPH_PLAIDML_ENABLE}")
message(STATUS "NGRAPH_TARGET_ARCH:         ${NGRAPH_TARGET_ARCH}")
message(STATUS "NGRAPH_TUNE_ARCH:           ${NGRAPH_TUNE_ARCH}")

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    add_definitions(-DNGRAPH_DISTRIBUTED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()
# Find and build ngraph - if not using pre-built one
if (NOT USE_PRE_BUILT_NGRAPH)
    ExternalProject_Add(
        ext_ngraph
        GIT_REPOSITORY https://github.com/NervanaSystems/ngraph
        GIT_TAG v0.27.1-rc.1
        CMAKE_ARGS 
            -DNGRAPH_DISTRIBUTED_ENABLE=${NGRAPH_DISTRIBUTED_ENABLE}
            -DNGRAPH_INSTALL_PREFIX=${NGRAPH_ARTIFACTS_DIR}
            -DNGRAPH_USE_CXX_ABI=${TensorFlow_CXX_ABI}
            -DNGRAPH_UNIT_TEST_ENABLE=FALSE
            -DNGRAPH_TOOLS_ENABLE=${NGRAPH_TOOLS_ENABLE}
            -DNGRAPH_DEX_ONLY=TRUE
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DNGRAPH_GPU_ENABLE=${NGRAPH_GPU_ENABLE}
            -DNGRAPH_DEBUG_ENABLE=${NGRAPH_DEBUG_ENABLE}
            -DNGRAPH_PLAIDML_ENABLE=${NGRAPH_PLAIDML_ENABLE}
            -DNGRAPH_TARGET_ARCH=${NGRAPH_TARGET_ARCH}
            -DNGRAPH_TUNE_ARCH=${NGRAPH_TUNE_ARCH}
        TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/tmp"
        STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/stamp"
        DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/download"
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/src"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ngraph/build"
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j ${NUM_MAKE_PROCESSES} ngraph
        UPDATE_COMMAND ""
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
    )
endif()
set(NGRAPH_INSTALL_DIR ${NGRAPH_ARTIFACTS_DIR})

if(NGRAPH_BRIDGE_STATIC_LIB_ENABLE)
    add_definitions(-DNGRAPH_BRIDGE_STATIC_LIB_ENABLE)
endif()

if(OS_VERSION STREQUAL "\"centos\"")
    set(LIB "lib64")
else()
    set(LIB "lib")
endif()

set(NGRAPH_IMPORTED_LOCATION ${NGRAPH_INSTALL_DIR}/${LIB}/${LIBNGRAPH})

if(NGRAPH_BRIDGE_STATIC_LIB_ENABLE)
    set(CPU_BACKEND_IMPORTED_LOCATION_STATIC ${NGRAPH_INSTALL_DIR}/${LIB}/${LIBCPU_BACKEND_A})
    set(INTERPRETER_BACKEND_IMPORTED_LOCATION_STATIC ${NGRAPH_INSTALL_DIR}/${LIB}/${LIBINTERPRETER_BACKEND_A})
    set(LIB_MKLDNN_IMPORTED_LOCATION ${NGRAPH_INSTALL_DIR}/${LIB}/libmkldnn.so)
    set(LIB_MKLML_INTEL_IMPORTED_LOCATION ${NGRAPH_INSTALL_DIR}/${LIB}/libmklml_intel.so)
    set(LIB_IOMP5_IMPORTED_LOCATION ${NGRAPH_INSTALL_DIR}/${LIB}/libiomp5.so)

    add_library(ngraph_lib STATIC IMPORTED)
    set_target_properties(
        ngraph_lib
        PROPERTIES IMPORTED_LOCATION
        ${NGRAPH_IMPORTED_LOCATION}
    )   

    add_library(lib_cpu_backend_static STATIC IMPORTED)
    set_target_properties(
        lib_cpu_backend_static
        PROPERTIES IMPORTED_LOCATION
        ${CPU_BACKEND_IMPORTED_LOCATION_STATIC}
    )

    add_library(lib_interpreter_backend_static STATIC IMPORTED)
    set_target_properties(
        lib_interpreter_backend_static
        PROPERTIES IMPORTED_LOCATION
        ${INTERPRETER_BACKEND_IMPORTED_LOCATION_STATIC}
    )

    add_library(lib_mkldnn SHARED IMPORTED)
    set_target_properties(
        lib_mkldnn
        PROPERTIES IMPORTED_LOCATION
        ${LIB_MKLDNN_IMPORTED_LOCATION}
    )

    add_library(lib_mklml_intel SHARED IMPORTED)
    set_target_properties(
        lib_mklml_intel
        PROPERTIES IMPORTED_LOCATION
        ${LIB_MKLML_INTEL_IMPORTED_LOCATION}
    )

    add_library(lib_iomp5 SHARED IMPORTED)
    set_target_properties(
        lib_iomp5
        PROPERTIES IMPORTED_LOCATION
        ${LIB_IOMP5_IMPORTED_LOCATION}
    )

else()
    add_library(ngraph_lib SHARED IMPORTED)
    set_target_properties(
        ngraph_lib
        PROPERTIES IMPORTED_LOCATION
        ${NGRAPH_IMPORTED_LOCATION}
    )
endif()

add_dependencies(ngraph_lib ext_ngraph)

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")

# Add the directories to be built
add_subdirectory(third-party)
add_subdirectory(logging)
add_subdirectory(ngraph_bridge)
add_subdirectory(tools)

# The following targets depend on the Tensorflow source code directory
# Get the absolute file name for the source
get_filename_component(
    TensorFlow_SRC_DIR
    "${TF_SRC_DIR}"
    ABSOLUTE
)

add_subdirectory(experimental/ngraph_device)
add_subdirectory(examples)
if (DEFINED TF_SRC_DIR)
    message(STATUS "TensorFlow_SRC_DIR: ${TensorFlow_SRC_DIR}")
    add_subdirectory(examples/cpp)
else()
    message(
        STATUS 
        "TensorFlow source directory not provided. "
        "C++ Examples won't be built"
    )
endif()

if (UNIT_TEST_ENABLE)
    if (NOT DEFINED TF_SRC_DIR)
        message(FATAL_ERROR "Provide TensorFlow source directory: -DTF_SRC_DIR=<directory>")
    endif()
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

    add_subdirectory(test)
    message(STATUS "unit tests enabled")
endif()

