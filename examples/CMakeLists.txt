# ******************************************************************************
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mnist)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/mnist/mnist_deep_simplified.py
        ${CMAKE_CURRENT_BINARY_DIR}/mnist/mnist_deep_simplified.py
)

