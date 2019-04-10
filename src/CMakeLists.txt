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

set(LIB_NAME ngraph_bridge)

include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)
include_directories(${NGRAPH_INSTALL_DIR}/include)
include_directories(${ABSL_INCLUDE_DIR})

set(NGRAPH_DEVICE_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason the following is needed for ABI compatibility with TF.
# (There must be some dependency on it in the struct/class definitions.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
set(SRC 
   ngraph_api.cc
   ngraph_assign_clusters.cc
   ngraph_builder.cc
   ngraph_backend_manager.cc
   ngraph_capture_variables.cc
   ngraph_cluster_manager.cc
   ngraph_deassign_clusters.cc
   ngraph_encapsulate_clusters.cc
   ngraph_encapsulate_op.cc
   ngraph_freshness_tracker.cc
   ngraph_mark_for_clustering.cc
   ngraph_rewrite_for_tracking.cc
   ngraph_rewrite_pass.cc
   ngraph_tracked_variable.cc
   ngraph_utils.cc
   tf_graphcycles.cc
   tf_deadness_analysis.cc
   version.cc
)

message(STATUS "NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS: ${NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS}")

if(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    # common files
    list(REMOVE_ITEM SRC ngraph_capture_variables.cc)
    list(APPEND SRC enable_variable_ops/ngraph_capture_variables.cc)

    list(REMOVE_ITEM SRC ngraph_encapsulate_op.cc)
    list(APPEND SRC enable_variable_ops/ngraph_encapsulate_op.cc)

    list(REMOVE_ITEM SRC ngraph_rewrite_for_tracking.cc)
    list(APPEND SRC enable_variable_ops/ngraph_rewrite_for_tracking.cc)

    list(REMOVE_ITEM SRC ngraph_rewrite_pass.cc)
    list(APPEND SRC enable_variable_ops/ngraph_rewrite_pass.cc)

    list(REMOVE_ITEM SRC ngraph_tracked_variable.cc)
    list(APPEND SRC enable_variable_ops/ngraph_tracked_variable.cc)

    list(REMOVE_ITEM SRC ngraph_utils.cc)
    list(APPEND SRC enable_variable_ops/ngraph_utils.cc)

    # new files
    list(APPEND SRC enable_variable_ops/ngraph_assign_op.cc)
    list(APPEND SRC enable_variable_ops/ngraph_catalog.cc)
    list(APPEND SRC enable_variable_ops/ngraph_enter_in_catalog.cc)
    list(APPEND SRC enable_variable_ops/ngraph_replace_op_utilities.cc)
    list(APPEND SRC enable_variable_ops/ngraph_replace_variable_modifiers.cc)
    list(APPEND SRC enable_variable_ops/ngraph_variable_modifiers.cc)

endif()


if(NGRAPH_TF_USE_GRAPPLER_OPTIMIZER)            
    list(REMOVE_ITEM SRC ngraph_rewrite_pass.cc)
    list(APPEND SRC grappler/ngraph_optimizer.cc)
    add_definitions(-DNGRAPH_TF_USE_GRAPPLER_OPTIMIZER)
endif()

message(STATUS "NGRAPH_TF_USE_GRAPPLER_OPTIMIZER: ${NGRAPH_TF_USE_GRAPPLER_OPTIMIZER}")

add_library(${LIB_NAME} SHARED ${SRC})

target_link_libraries( ${LIB_NAME} ngraph_logger) 
target_link_libraries( ${LIB_NAME} tensorflow_framework_lib )
target_link_libraries( ${LIB_NAME} ngraph_lib )
target_link_libraries( ${LIB_NAME} absl_algorithm )
target_link_libraries( ${LIB_NAME} absl_container )
target_link_libraries( ${LIB_NAME} absl_strings )

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

if(OS_VERSION STREQUAL "\"centos\"")
	set(NGTF_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib64)
else()
	set(NGTF_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()

message(STATUS "NGTF_INSTALL_LIB_DIR: ${NGTF_INSTALL_LIB_DIR}")

# First install the libngraph_bridge.so and headers
install(TARGETS ${LIB_NAME} DESTINATION "${NGTF_INSTALL_LIB_DIR}")  
install(FILES ngraph_backend_manager.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)


# Next install all the other prerequisites
if(OS_VERSION STREQUAL "\"centos\"")
	install(DIRECTORY ${NGRAPH_INSTALL_DIR}/lib64/ DESTINATION "${NGTF_INSTALL_LIB_DIR}" )
else()
	install(DIRECTORY ${NGRAPH_INSTALL_DIR}/lib/ DESTINATION "${NGTF_INSTALL_LIB_DIR}" )
endif()

#Install nbench if we built
if(DEFINED NGRAPH_TOOLS_ENABLE)
	install(DIRECTORY ${NGRAPH_INSTALL_DIR}/bin/ DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" )
endif()

# Now create the Python pip package. The following variables are passed to the 
# CreatePipWhl.cmake
# CMake "install" target "CODE" will propagate these variables to the 
# target "SCRIPT" so that the "SCRIPT" (in ths case CreatePipWhl.cmake)
# So any variable that we need to propagate needs to be added here
install(CODE "set(OS_VERSION ${OS_VERSION})")
install(CODE "set(NGRAPH_INSTALL_DIR \"${NGRAPH_INSTALL_DIR}\")")
install(CODE "set(NGTF_SRC_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/../\")")
install(CODE "set(NGTF_INSTALL_DIR \"${CMAKE_INSTALL_PREFIX}\")")
install(CODE "set(TensorFlow_GIT_VERSION \"${TensorFlow_GIT_VERSION}\")")
install(CODE "set(TensorFlow_VERSION \"${TensorFlow_VERSION}\")")
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../python/CreatePipWhl.cmake")
