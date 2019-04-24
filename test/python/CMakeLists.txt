# Copyright 2018-2019 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.1)

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

