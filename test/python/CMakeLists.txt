# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")
list(APPEND files "tox.ini" ".pylintrc")
foreach(file ${files})
    execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/${file}
        ${CMAKE_CURRENT_BINARY_DIR}/${file}
)
endforeach()

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/flib_graph_1.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/flib_graph_1.pbtxt
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/flib_graph_2.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/flib_graph_2.pbtxt
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/sample_graph.pb
        ${CMAKE_CURRENT_BINARY_DIR}/sample_graph.pb
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/sample_graph.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/sample_graph.pbtxt
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/sample_graph
        ${CMAKE_CURRENT_BINARY_DIR}/sample_graph
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/sample_graph_nodevice.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/sample_graph_nodevice.pbtxt
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/fusedmatmul_0.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/fusedmatmul_0.pbtxt
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/fusedmatmul_1.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/fusedmatmul_1.pbtxt
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/fusedmatmul_2.pbtxt
        ${CMAKE_CURRENT_BINARY_DIR}/fusedmatmul_2.pbtxt
)
