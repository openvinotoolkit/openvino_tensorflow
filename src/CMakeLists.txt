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

include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)
include_directories(${NGRAPH_DIR}/include)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${TensorFlow_DIR}/plugins/ngraph/lib")

add_library(tensorflow_xla_plugin_framework SHARED IMPORTED)
set_target_properties( 
    tensorflow_xla_plugin_framework 
    PROPERTIES IMPORTED_LOCATION 
    ${TensorFlow_DIR}/libtensorflow_framework.so
)

add_library(ngraph_lib SHARED IMPORTED)
set_target_properties( 
    ngraph_lib 
    PROPERTIES IMPORTED_LOCATION 
    ${NGRAPH_DIR}/lib/libngraph.so
)
add_dependencies(ngraph_lib ext_ngraph)

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
set(SRC 
   ngraph_plugin.cc 
   transfer_manager.cc
   ngraph_emitter.cc
   ngraph_compiler.cc
   ngraph_autobroadcast.cc
   ngraph_log.cc
   ngraph_utils.cc
   ngraph_fusion.cc
   ngraph_executable.cc
   ngraph_xla_compat.cc
)

add_library(${NGRAPH_TENSORFLOW_BRIDGE_LIB_NAME} SHARED ${SRC})
target_link_libraries( 
    ${NGRAPH_TENSORFLOW_BRIDGE_LIB_NAME} 
    tensorflow_xla_plugin_framework 
)

target_link_libraries( 
    ${NGRAPH_TENSORFLOW_BRIDGE_LIB_NAME} 
    ngraph_lib 
)

target_compile_definitions( 
    ${NGRAPH_TENSORFLOW_BRIDGE_LIB_NAME} PRIVATE 
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI} 
)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/cmake_uninstall.in.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# Installation
# Ordering is key here as during the uninslall we will follow this
# order to remove the directories that were installed

# First install the dependencies
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../ngraph/ngraph_dist/lib
    DESTINATION 
    plugins/ngraph
)

# Next install the plugin library
install(
    TARGETS 
    ${NGRAPH_TENSORFLOW_BRIDGE_LIB_NAME} 
    LIBRARY DESTINATION plugins
)  
