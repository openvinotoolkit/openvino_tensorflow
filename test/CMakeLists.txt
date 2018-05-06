# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include_directories(${GTEST_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)

set(TensorFlow_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../ngraph-tensorflow")
include_directories(${TensorFlow_SRC_DIR})

# include_directories(
#     ${NGRAPH_INCLUDE_DIR}
#     )

set(SRC
    main.cpp
    graph_exec.cpp
)

set(TF_OP_DEF_SRC
    ${TensorFlow_SRC_DIR}/tensorflow/core/ops/array_ops.cc
    ${TensorFlow_SRC_DIR}/tensorflow/core/ops/sendrecv_ops.cc
    ${TensorFlow_SRC_DIR}/tensorflow/core/ops/math_ops.cc
    ${TensorFlow_SRC_DIR}/tensorflow/core/ops/no_op.cc
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(gtest_ngtf ${SRC} ${TF_OP_DEF_SRC})
target_link_libraries(gtest_ngtf ngraph_device libgtest pthread tensorflow_framework_lib)
#add_dependencies(gtest_ngtf libgtest tensorflow_framework_lib)
