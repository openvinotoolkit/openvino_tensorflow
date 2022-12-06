# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME tf_conversion_extensions)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create library
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})

target_include_directories(${TARGET_NAME}
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}
            ${InferenceEngine_INCLUDE_DIRS}
            ${OV_FRONTEND_INCLUDE_DIRECTORIES}
            ${OV_RUNTIME_INCLUDE_DIRECTORIES}
            ${OV_RUNTIME_C_INCLUDE_DIRECTORIES}
            ${OPENVINO_INSTALL_DIR}/include/)

target_link_libraries(${TARGET_NAME} tensorflow_frontend_lib openvino_lib)

if ((STANDALONE_CMAKE) AND (NOT USE_OPENVINO_FROM_LOCATION))
  add_dependencies(${TARGET_NAME}
                   openvino::runtime
                   openvino::runtime::c
                   openvino::frontend::tensorflow)
endif()

install(CODE "set(TF_CONVERSION_EXTENSIONS_LIB_NAME \"${TARGET_NAME}\")")
