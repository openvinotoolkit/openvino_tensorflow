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

set(LIB_NAME ngraph_device)

include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)
include_directories(${NGRAPH_INSTALL_DIR}/include)

set(NGRAPH_DEVICE_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason the following is needed for ABI compatibility with TF.
# (There must be some dependency on it in the struct/class definitions.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

if(APPLE)
    set(LIBNGRAPH_SO "libngraph.dylib")
else()
    set(LIBNGRAPH_SO "libngraph.so")
endif(APPLE)

add_library(ngraph_lib SHARED IMPORTED)
set_target_properties( 
    ngraph_lib 
    PROPERTIES IMPORTED_LOCATION 
    ${NGRAPH_INSTALL_DIR}/lib/${LIBNGRAPH_SO}
)
add_dependencies(ngraph_lib ext_ngraph)

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
set(SRC 
   ngraph_builder.cc
   ngraph_cluster.cc
   ngraph_cluster_manager.cc
   ngraph_confirm_pass.cc
   ngraph_device.cc
   ngraph_encapsulate_op.cc
   ngraph_encapsulate_pass.cc
   ngraph_freshness_tracker.cc
   ngraph_graph_rewrite_passes.cc
   ngraph_liberate_pass.cc
   ngraph_mark_for_hostmem_pass.cc
   ngraph_op_kernels.cc
   ngraph_stub_ops.cc
   ngraph_utils.cc
   tf_graphcycles.cc
   ngraph_send_recv_ops.cc
   ngraph_variable_ops.cc
)

add_library(${LIB_NAME} SHARED ${SRC})

target_link_libraries( ${LIB_NAME} ngraph_logger) 
target_link_libraries( ${LIB_NAME} tensorflow_framework_lib )
target_link_libraries( ${LIB_NAME} ngraph_lib )

target_compile_definitions( 
    ${LIB_NAME} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}
)

target_include_directories(${LIB_NAME} PUBLIC "${NGRAPH_INSTALL_DIR}/include")
target_include_directories(${LIB_NAME} PUBLIC "${NGRAPH_DEVICE_INCLUDE_PATH}")

#------------------------------------------------------------------------------
#installation 
#------------------------------------------------------------------------------
if (DEFINED NGRAPH_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()

# First install the libngraph_device.so
install(TARGETS ngraph_device DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")  

# Next install all the other prerequisites
install(DIRECTORY ${NGRAPH_INSTALL_DIR}/lib/ DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" )

# Now create the Python pip package
install(CODE "set(NGTF_INSTALL_DIR \"${CMAKE_INSTALL_PREFIX}/lib\")")
install(CODE "set(TensorFlow_GIT_VERSION \"${TensorFlow_GIT_VERSION}\")")
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../python/CreatePipWhl.cmake")
