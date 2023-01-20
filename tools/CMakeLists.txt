# ******************************************************************************
# Copyright (C) 2023 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

if (DEFINED OPENVINO_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${OPENVINO_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()


# copy to build_cmake/artifacts/tools
install(
   FILES ${CMAKE_CURRENT_LIST_DIR}/build_utils.py
   DESTINATION ${CMAKE_INSTALL_PREFIX}/tools
)
install(
   FILES ${CMAKE_CURRENT_LIST_DIR}/log_parser.py
   DESTINATION ${CMAKE_INSTALL_PREFIX}/tools
)

# copies to build_cmake/tools
file(COPY ${CMAKE_CURRENT_LIST_DIR}/build_utils.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/log_parser.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

