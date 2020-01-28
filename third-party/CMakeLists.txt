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

cmake_minimum_required(VERSION 3.4)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()
include( ../cmake/external_gtest.cmake )

ExternalProject_Add(
    ext_abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG f21d187b80e3b7f08fb279775ea9c8b48c636030
    CMAKE_ARGS
        -DBUILD_TESTING=OFF
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=-fPIC
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/abseil/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/abseil/build"
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL
)

get_filename_component(
    ABSL_INCLUDE_DIR
    "${EXTERNAL_PROJECTS_ROOT}/abseil/src"
    ABSOLUTE
)
set(ABSL_INCLUDE_DIR "${ABSL_INCLUDE_DIR}" PARENT_SCOPE)

get_filename_component(
    ABSL_BUILD_DIR
    "${EXTERNAL_PROJECTS_ROOT}/abseil/build/absl"
    ABSOLUTE
)
set(ABSL_BUILD_DIR "${ABSL_BUILD_DIR}")

# Create the absl libraries - algorithm
add_library(absl_algorithm IMPORTED STATIC GLOBAL)
add_dependencies(absl_algorithm ext_abseil)

# Set library properties
set_target_properties(absl_algorithm PROPERTIES
    "IMPORTED_LOCATION" "${ABSL_BUILD_DIR}/algorithm/libabsl_algorithm.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create the absl libraries - container
add_library(absl_container IMPORTED STATIC GLOBAL)
add_dependencies(absl_container ext_abseil)

# Set library properties
set_target_properties(absl_container PROPERTIES
    "IMPORTED_LOCATION" "${ABSL_BUILD_DIR}/container/libabsl_container.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create the absl libraries - strings
add_library(absl_strings IMPORTED STATIC GLOBAL)
add_dependencies(absl_strings ext_abseil)

# Set library properties
set_target_properties(absl_strings PROPERTIES
    "IMPORTED_LOCATION" "${ABSL_BUILD_DIR}/strings/libabsl_strings.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create the absl libraries - synchronization
add_library(absl_synchronization IMPORTED STATIC GLOBAL)
add_dependencies(absl_synchronization ext_abseil)

# Set library properties
set_target_properties(absl_synchronization PROPERTIES
    "IMPORTED_LOCATION" "${ABSL_BUILD_DIR}/synchronization/libabsl_synchronization.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
