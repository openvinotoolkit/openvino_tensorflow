# Copyright (C) 2021-2022 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0

set(LIB_NAME ovtf_logger)

# For some reason the following is needed for ABI compatibility with TF.
# (There must be some dependency on it in the struct/class definitions.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -std=c++17")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
set(SRC 
   ovtf_log.cc
   tf_graph_writer.cc
)

add_library(${LIB_NAME} ${SRC})

target_include_directories(${LIB_NAME} PRIVATE ${TensorFlow_INCLUDE_DIR}
                           ${TensorFlow_INCLUDE_DIR}/external/nsync/public)
target_link_libraries( 
    ${LIB_NAME} 
    ${TensorFlow_FRAMEWORK_LIBRARY} 
)

target_include_directories(${LIB_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${LIB_NAME} PRIVATE "${TensorFlow_INCLUDE_DIR}")
target_include_directories(${LIB_NAME} PRIVATE "${TensorFlow_INCLUDE_DIR}/external/nsync/public")

target_compile_definitions( 
    ${LIB_NAME} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}
)
set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Installation
if (DEFINED OPENVINO_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${OPENVINO_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()

install(FILES ovtf_log.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
