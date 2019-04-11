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

# Create the pip package
find_program(PYTHON "python")

message( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_BINARY_DIR}")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_LIST_DIR}/setup.in.py")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/python/setup.py")
    set(INIT_PY_IN  "${CMAKE_CURRENT_LIST_DIR}/ngraph_bridge/__init__.in.py")
    set(INIT_PY     "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge/__init__.py")
    set(PIP_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/build_pip")

    # Set the readme document location
    get_filename_component(
        readme_file_path ${CMAKE_CURRENT_LIST_DIR}/../README.md ABSOLUTE)
    set(README_DOC \"${readme_file_path}\")

    # Create the python/ngraph_bridge directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/ngraph_bridge)

    # Get the list of libraries we need for the Python pip package
    # If we are building on CentOS then it's lib64 - else lib
    set(LIB_SUFFIX lib)
    if(NOT APPLE)
        if(OS_VERSION STREQUAL "centos")
            set(LIB_SUFFIX lib64)
        endif()
    endif()
    message(STATUS "LIB_SUFFIX: ${NGTF_INSTALL_DIR}/${LIB_SUFFIX}")
    file(GLOB NGRAPH_LIB_FILES "${NGTF_INSTALL_DIR}/${LIB_SUFFIX}/lib*")

    # Copy the ngraph_bridge include from install
    message(STATUS "NGTF_INSTALL_DIR: ${NGTF_INSTALL_DIR}")
    
    file(
        COPY "${NGRAPH_INSTALL_DIR}/include" 
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge"
    )

    # Copy the ngraph_bridge libraries from install
    foreach(DEP_FILE ${NGRAPH_LIB_FILES})
        get_filename_component(lib_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lib_file_name ${DEP_FILE} NAME)
        set(ngraph_libraries "${ngraph_libraries}\"${lib_file_name}\",\n\t")
        file(COPY ${lib_file_real_path}
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge")
    endforeach()

    # Get the list of license files
    file(GLOB NGRAPH_TF_LICENSE_FILES "${NGTF_SRC_DIR}/third-party/licenses/*")

    # Copy the licenses for ngraph-tf
    foreach(DEP_FILE ${NGRAPH_TF_LICENSE_FILES})
        get_filename_component(lic_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lic_file_name ${DEP_FILE} NAME)
        set(
            license_files
            "${license_files}\"licenses/${lic_file_name}\",\n\t")
        file(COPY ${lic_file_real_path}
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge/licenses")
    endforeach()

    # Get the list of license files for ngraph
    file(GLOB NGRAPH_LICENSE_FILES "${NGRAPH_INSTALL_DIR}/licenses/*")

    # Copy the licenses for ngraph-tf
    foreach(DEP_FILE ${NGRAPH_LICENSE_FILES})
        get_filename_component(lic_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lic_file_name ${DEP_FILE} NAME)
        set(
            license_files
            "${license_files}\"licenses/${lic_file_name}\",\n\t")
        file(COPY ${lic_file_real_path}
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge/licenses")
    endforeach()

    # Copy the LICENSE at the top level
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../LICENSE
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge")
    set(
        licence_top_level
        "\"LICENSE\"")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})
    configure_file(${INIT_PY_IN} ${INIT_PY})
    if (APPLE)
        execute_process(COMMAND
            install_name_tool -change
            libngraph.dylib
            @loader_path/libngraph.dylib
            ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge/libngraph_bridge.dylib
            RESULT_VARIABLE result
            ERROR_VARIABLE ERR
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(${result})
            message(FATAL_ERROR "Cannot update @loader_path")
        endif()

        execute_process(COMMAND
            install_name_tool -change
            libngraph.dylib
            @loader_path/libngraph.dylib
            ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge/libcpu_backend.dylib
            RESULT_VARIABLE result
            ERROR_VARIABLE ERR
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(${result})
            message(FATAL_ERROR "Cannot update @loader_path")
        endif()

        set(cpu_lib_list
            libmkldnn.0.dylib
            libmklml.dylib
            libiomp5.dylib
            libtbb.dylib
        )

        FOREACH(lib_file ${cpu_lib_list})
            message("Library: " ${lib_file})
            execute_process(COMMAND
                install_name_tool -change
                @rpath/${lib_file}
                @loader_path/${lib_file}
                ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_bridge/libcpu_backend.dylib
                RESULT_VARIABLE result
                ERROR_VARIABLE ERR
                ERROR_STRIP_TRAILING_WHITESPACE
            )
        ENDFOREACH()

    endif()

    execute_process(
        COMMAND ${PYTHON} "setup.py" "bdist_wheel"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/
    )

endif()
