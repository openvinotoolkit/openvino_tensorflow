# Copyright 2018-2020 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(LIB_NAME ngraph_bridge_device)
set(LIB_NAME_STATIC ngraph_bridge_device_static)

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
   ngraph_device.cc
   ngraph_ops.cc
   ngraph_send_recv_ops.cc
   ngraph_optimization_pass.cc
)

add_library(${LIB_NAME} SHARED ${SRC})

if(NGRAPH_BRIDGE_STATIC_LIB_ENABLE)
    target_link_libraries(
            ${LIB_NAME}
            ngraph_logger 
            ${TensorFlow_FRAMEWORK_LIBRARY} 
            ngraph_lib
            lib_cpu_backend_static
            lib_interpreter_backend_static 
            ngraph_lib
            dl
            absl_algorithm
            absl_container
            absl_strings
            lib_mklml_intel
            lib_dnnl
            lib_iomp5
        )

        target_compile_definitions(
        ${LIB_NAME} PRIVATE 
        "NGRAPH_BRIDGE_STATIC_LIB_ENABLE")
else()
    target_link_libraries(
            ${LIB_NAME}
            ngraph_logger 
            ${TensorFlow_FRAMEWORK_LIBRARY} 
            ngraph_lib
            absl_algorithm
            absl_container
            absl_strings
        )
endif()

target_compile_definitions( 
    ${LIB_NAME} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}
)

target_include_directories(${LIB_NAME} PUBLIC "${NGRAPH_INSTALL_DIR}/include")
target_include_directories(${LIB_NAME} PUBLIC "${NGRAPH_DEVICE_INCLUDE_PATH}")

if(NGRAPH_BRIDGE_STATIC_LIB_ENABLE)
    add_library(${LIB_NAME_STATIC} STATIC ${SRC})

    target_link_libraries(
        ${LIB_NAME_STATIC}
        ngraph_logger
        ${TensorFlow_FRAMEWORK_LIBRARY} 
        ngraph_lib
        lib_cpu_backend_static
        lib_interpreter_backend_static 
        ngraph_lib
        dl
        absl_algorithm
        absl_container
        absl_strings
        lib_dnnl
        lib_iomp5
        lib_mklml_intel
    )
    target_compile_definitions(
    ${LIB_NAME_STATIC} PRIVATE 
    "NGRAPH_BRIDGE_STATIC_LIB_ENABLE")
    
    target_compile_definitions( 
    ${LIB_NAME_STATIC} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}
    )

    target_include_directories(${LIB_NAME_STATIC} PUBLIC "${NGRAPH_INSTALL_DIR}/include")
    target_include_directories(${LIB_NAME_STATIC} PUBLIC "${NGRAPH_DEVICE_INCLUDE_PATH}")

endif()

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

file(GLOB test_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")
foreach(file ${test_files})
    execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/${file}
        ${CMAKE_CURRENT_BINARY_DIR}/${file}
)
endforeach()
