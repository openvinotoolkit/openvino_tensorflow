# ******************************************************************************
# Copyright 2019-2020 Intel Corporation
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

if (DEFINED NGRAPH_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_TF_INSTALL_PREFIX})
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

