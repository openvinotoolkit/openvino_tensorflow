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
    set(INIT_PY_IN  "${CMAKE_CURRENT_LIST_DIR}/ngraph_config/__init__.in.py")
    set(INIT_PY     "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/__init__.py")
    set(PIP_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/build_pip")

    # Create the python/ngraph_config directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/ngraph_config)

    # Get the list of libraries we need for the Python pip package
    file(GLOB NGRAPH_LIB_FILES "${NGTF_INSTALL_DIR}/lib/lib*")
    
    # Copy the ngraph_config libraries from install
    foreach(DEP_FILE ${NGRAPH_LIB_FILES})
        get_filename_component(lib_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lib_file_name ${DEP_FILE} NAME)
        set(ngraph_libraries "${ngraph_libraries}\"${lib_file_name}\",\n\t")
        file(COPY ${lib_file_real_path} 
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config")        
    endforeach()            

    # Get the list of license files 
    file(GLOB NGRAPH_TF_LICENCE_FILES "${NGTF_SRC_DIR}/third-party/licenses/*")
    
    # Copy the licenses for ngraph-tf
    foreach(DEP_FILE ${NGRAPH_TF_LICENCE_FILES})
    message(STATUS "Lic file: ${DEP_FILE}")
        get_filename_component(lic_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lic_file_name ${DEP_FILE} NAME)
    message(STATUS "lic_file_real_path ${lic_file_real_path}")
        set(
            license_files 
            "${license_files}\"licenses/${lic_file_name}\",\n\t")
        file(COPY ${lic_file_real_path} 
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/licenses")        
    endforeach()            

    # Get the list of license files for ngraph
    file(GLOB NGRAPH_LICENCE_FILES "${NGRAPH_INSTALL_DIR}/licenses/*")
    
    # Copy the licenses for ngraph-tf
    foreach(DEP_FILE ${NGRAPH_LICENCE_FILES})
    message(STATUS "Lic file: ${DEP_FILE}")
        get_filename_component(lic_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lic_file_name ${DEP_FILE} NAME)
    message(STATUS "lic_file_real_path ${lic_file_real_path}")
        set(
            license_files 
            "${license_files}\"licenses/${lic_file_name}\",\n\t")
        file(COPY ${lic_file_real_path} 
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/licenses")        
    endforeach()            

    # Copy the LICENSE at the toplevel
    file(COPY ${CMAKE_SOURCE_DIR}/../LICENSE 
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config")        
    set(
        licence_top_level 
        "\"LICENSE\"")

    message(STATUS "NGRAPH_INSTALL_DIR ${NGRAPH_INSTALL_DIR}")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})
    configure_file(${INIT_PY_IN} ${INIT_PY})
    if (APPLE)
        execute_process(COMMAND 
            install_name_tool -change 
            libngraph.dylib 
            @loader_path/libngraph.dylib 
            ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/libngraph_bridge.dylib
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
            ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/libcpu_backend.dylib
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
                ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/libcpu_backend.dylib
                RESULT_VARIABLE result
                ERROR_VARIABLE ERR
                ERROR_STRIP_TRAILING_WHITESPACE
            )
        ENDFOREACH()

        if ("${NGRAPH_LIB_FILES};" MATCHES "/libplaidml_backend.dylib;")
            execute_process(COMMAND
                install_name_tool -change
                libngraph.dylib
                @loader_path/libngraph.dylib
                ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/libplaidml_backend.dylib
                RESULT_VARIABLE result
                ERROR_VARIABLE ERR
                ERROR_STRIP_TRAILING_WHITESPACE
            )
            if(${result})
                message(FATAL_ERROR "Cannot update @loader_path")
            endif()

            execute_process(COMMAND
                install_name_tool -change
                libplaidml.dylib
                @loader_path/libplaidml.dylib
                ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/libplaidml_backend.dylib
                RESULT_VARIABLE result
                ERROR_VARIABLE ERR
                ERROR_STRIP_TRAILING_WHITESPACE
            )
            if(${result})
                message(FATAL_ERROR "Cannot update @loader_path")
            endif()

            set(plaidml_lib_list
                libplaidml.dylib
            )

            FOREACH(lib_file ${plaidml_lib_list})
                message("Library: " ${lib_file})
                execute_process(COMMAND
                    install_name_tool -change
                    @rpath/${lib_file}
                    @loader_path/${lib_file}
                    ${CMAKE_CURRENT_BINARY_DIR}/python/ngraph_config/libplaidml_backend.dylib
                    RESULT_VARIABLE result
                    ERROR_VARIABLE ERR
                    ERROR_STRIP_TRAILING_WHITESPACE
                )
            ENDFOREACH()
        endif()

    endif()

    execute_process(
        COMMAND ${PYTHON} "setup.py" "bdist_wheel"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/
    )
    
endif()
