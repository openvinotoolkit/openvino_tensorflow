# ******************************************************************************
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

add_subdirectory(python)

if (DEFINED OPENVINO_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${OPENVINO_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/python 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/test
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pbtxt"
    PATTERN "*.pb"
    PATTERN "tests_*.txt"
)
# C++ gtests are not enabled for Windows yet
if (WIN32)
    return()
endif()

# Get this location from the CMAKE variables passed during the cmake time
if (NOT DEFINED TF_SRC_DIR)
   message(
       STATUS
       "TensorFlow source directory not found. Skipping C++ unit tests..."
    )
    return()
endif()

include_directories(${GTEST_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)
include_directories(${TensorFlow_SRC_DIR})

# Files that are generated during TF build are here
include_directories(${TensorFlow_SRC_DIR}/bazel-genfiles)
include_directories(${TensorFlow_SRC_DIR}/bazel-bin)

add_library(tensorflow_cc_lib SHARED IMPORTED)
add_library(tensorflow_ops_testutil SHARED IMPORTED)

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

set_target_properties(
    tensorflow_ops_testutil
    PROPERTIES IMPORTED_LOCATION
    ${TensorFlow_SRC_DIR}/bazel-bin/tensorflow/core/kernels/libops_testutil.so
)

set(SRC
    main.cpp
    test_ovtf_exec.cpp
    tf_exec.cpp
    padding.cpp
    conversions.cpp
    graph_rewrites/assign_clusters.cc
    # graph_rewrites/deadness_test.cc
    graph_rewrites/backend_manager_test.cc
    graph_rewrites/encapsulate_clusters_test.cc
    # graph_rewrites/disable_ops_test.cc
    # graph_rewrites/mark_for_clustering_test.cc
    # graph_rewrites/op_by_op_capability_test.cc
    test_utilities.cpp
    test_math_ops.cpp
    test_nn_ops.cpp
    test_array_ops.cpp
    opexecuter.cpp
    test_thread_safe_queue.cc
    pass/transpose_sinking_test.cpp
)

if(OPENVINO_TF_USE_GRAPPLER_OPTIMIZER)
    list(APPEND SRC graph_rewrites/config_for_grappler_test.cc)
endif()

# The compile flag -DNDEBUG is required since
# tensorflow::Core::RefCounted is error prone as explained here:
# https://github.com/tensorflow/tensorflow/issues/17316
# Also, the openvino_tensorflow code when compiled has this flag enabled(https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/openvino_tensorflow/CMakeLists.txt#L23)
# hence the error seen while testing the custom op does not show up there.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DNDEBUG")

message(STATUS "Compile Flags for test dir: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")

add_executable(gtest_ovtf ${SRC})
add_dependencies(gtest_ovtf ext_gtest)
message(STATUS "TensorFlow_SRC_DIR: ${TensorFlow_SRC_DIR}")

# The following custom commands are used to create symlinks for various
# input files needed by the unit test
add_custom_command(
    TARGET gtest_ovtf
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy.pbtxt
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_launchop.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy_launchop.pbtxt
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_int8_launchop.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy_int8_launchop.pbtxt
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_8bit.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/test_axpy_8bit.pbtxt
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/test_general_graph.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/test_general_graph.pbtxt
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/test_graph1.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/test_graph1.pbtxt
)

if (APPLE)
    set(OPENVINO_TF_CXX11_ABI 0)
endif()

target_link_libraries(
    gtest_ovtf
    openvino_tensorflow
    libgtest
    pthread
    ${TensorFlow_FRAMEWORK_LIBRARY}
    tensorflow_cc_lib
    tensorflow_ops_testutil
    absl_synchronization
    ${InferenceEngine_LIBRARIES} ${TBB_IMPORTED_TARGETS}
    ocm
)

# First install the libopenvino_tensorflow.so and headers
install(TARGETS gtest_ovtf DESTINATION ${CMAKE_INSTALL_PREFIX}/test)  
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy.pbtxt DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_launchop.pbtxt DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_8bit.pbtxt DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_int8_launchop.pbtxt DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_general_graph.pbtxt DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_graph1.pbtxt DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
