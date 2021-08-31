# ******************************************************************************
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

# Try to find TensorFlow library and include paths
#
# The following are set after configuration is done:
#  TensorFlow_FOUND
#  TensorFlow_INCLUDE_DIR
#  TensorFlow_LIBRARIES
#  TensorFlow_LIBRARY_DIRS
#  TensorFlow_DIR

include(FindPackageHandleStandardArgs)
message(STATUS "Looking for TensorFlow installation")

execute_process(
    COMMAND
    python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE TensorFlow_INCLUDE_DIR
    ERROR_VARIABLE ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(${result})
    message(FATAL_ERROR "Cannot determine TensorFlow installation directory " ${ERR})
endif()
MESSAGE(STATUS "TensorFlow_INCLUDE_DIR: " ${TensorFlow_INCLUDE_DIR})

execute_process(
    COMMAND
    python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE TensorFlow_DIR
    ERROR_VARIABLE ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(${result})
    message(FATAL_ERROR "Cannot determine TensorFlow installation directory\n" ${ERR})
endif()
message(STATUS "TensorFlow_DIR: " ${TensorFlow_DIR})

execute_process(
    COMMAND
    python -c "import tensorflow as tf; print(tf.__cxx11_abi_flag__)"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE TensorFlow_CXX_ABI
    ERROR_VARIABLE ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(${result})
    message(FATAL_ERROR "Cannot determine TensorFlow __cxx11_abi_flag__\n" ${ERR})
endif()
message(STATUS "TensorFlow_CXX_ABI: " ${TensorFlow_CXX_ABI})

execute_process(
    COMMAND
    python -c "import tensorflow as tf; print(tf.__git_version__)"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE TensorFlow_GIT_VERSION
    ERROR_VARIABLE ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(${result})
    message(FATAL_ERROR "Cannot determine TensorFlow __git_version__\n" ${ERR})
endif()
message(STATUS "TensorFlow_GIT_VERSION: " ${TensorFlow_GIT_VERSION})

execute_process(
    COMMAND
    python -c "import tensorflow as tf; print(tf.__version__)"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE TensorFlow_VERSION
    ERROR_VARIABLE ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(${result})
    message(FATAL_ERROR "Cannot determine TensorFlow __version__\n" ${ERR})
endif()
message(STATUS "TensorFlow_VERSION: " ${TensorFlow_VERSION})

# Make sure that the TF library exists
if ( APPLE )
    if(NOT(TensorFlow_VERSION LESS 2.0))
        set(TF_LIB_NAME libtensorflow_framework.2.dylib)
    else()
        set(TF_LIB_NAME libtensorflow_framework.dylib)
    endif()
else()
    if(NOT(TensorFlow_VERSION LESS 2.0))
        set(TF_LIB_NAME libtensorflow_framework.so.2)
    else()
        set(TF_LIB_NAME libtensorflow_framework.so.1)
    endif()
endif()

message(STATUS "TF_LIB: " ${TF_LIB_NAME})

find_library(
  TensorFlow_FRAMEWORK_LIBRARY
  NAME ${TF_LIB_NAME}
  PATHS ${TensorFlow_DIR}
  NO_DEFAULT_PATH
)

find_package_handle_standard_args(
  TensorFlow
  FOUND_VAR TensorFlow_FOUND
  REQUIRED_VARS
    TensorFlow_DIR
    TensorFlow_INCLUDE_DIR
    TensorFlow_GIT_VERSION
    TensorFlow_VERSION
    TensorFlow_FRAMEWORK_LIBRARY
)
