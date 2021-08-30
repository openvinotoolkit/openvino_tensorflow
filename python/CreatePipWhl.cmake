# ******************************************************************************
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

# Create the pip package
if(PYTHON_EXECUTABLE)
    set(PYTHON ${PYTHON_EXECUTABLE})
else(PYTHON_EXECUTABLE)
    find_program(PYTHON "python")
endif()

message( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_BINARY_DIR}")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_LIST_DIR}/setup.in.py")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/python/setup.py")
    set(INIT_PY_IN  "${CMAKE_CURRENT_LIST_DIR}/openvino_tensorflow/__init__.in.py")
    set(INIT_PY     "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/__init__.py")
    set(PIP_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/build_pip")

    # Set the readme document location
    get_filename_component(
        readme_file_path ${CMAKE_CURRENT_LIST_DIR}/README.md ABSOLUTE)
    set(README_DOC \"${readme_file_path}\")

    # Create the python/openvino_tensorflow directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/openvino_tensorflow)

    # Get the list of libraries we need for the Python pip package
    # If we are building on CentOS then it's lib64 - else lib
    set(LIB_SUFFIX lib)
    if(NOT APPLE)
        if(OS_VERSION STREQUAL "centos")
            set(LIB_SUFFIX lib64)
        endif()
    endif()
    message(STATUS "LIB_SUFFIX: ${OVTF_INSTALL_DIR}/${LIB_SUFFIX}")
    file(GLOB NGRAPH_LIB_FILES "${OVTF_INSTALL_DIR}/${LIB_SUFFIX}/*")
    # Copy the openvino_tensorflow include from install
    message(STATUS "OVTF_INSTALL_DIR: ${OVTF_INSTALL_DIR}")

    # Copy the openvino_tensorflow libraries from install
    foreach(DEP_FILE ${NGRAPH_LIB_FILES})
        get_filename_component(lib_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lib_file_name ${DEP_FILE} NAME)
        set(ovtf_libraries "${ovtf_libraries}\"${lib_file_name}\",\n\t")
        file(COPY ${lib_file_real_path}
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow")
    endforeach()

    # Get the list of license files
    file(GLOB OPENVINO_TF_LICENSE_FILES "${NGTF_SRC_DIR}/third-party/licenses/*")

    # Copy the licenses for openvino-tensorflow
    foreach(DEP_FILE ${OPENVINO_TF_LICENSE_FILES})
        get_filename_component(lic_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lic_file_name ${DEP_FILE} NAME)
        set(
            license_files
            "${license_files}\"licenses/${lic_file_name}\",\n\t")
        file(COPY ${lic_file_real_path}
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/licenses")
    endforeach()

    # Get the list of license files for ngraph
    file(GLOB NGRAPH_LICENSE_FILES "${NGRAPH_INSTALL_DIR}/licenses/*")

    # Copy the licenses for openvino-tensorflow
    foreach(DEP_FILE ${NGRAPH_LICENSE_FILES})
        get_filename_component(lic_file_real_path ${DEP_FILE} ABSOLUTE)
        get_filename_component(lic_file_name ${DEP_FILE} NAME)
        set(
            license_files
            "${license_files}\"licenses/${lic_file_name}\",\n\t")
        file(COPY ${lic_file_real_path}
            DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/licenses")
    endforeach()

    # Copy the LICENSE at the top level
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../LICENSE
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow")
    set(
        licence_top_level
        "\"LICENSE\"")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})
    configure_file(${INIT_PY_IN} ${INIT_PY})
     if (APPLE)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(libMKLDNNPluginPath "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/libMKLDNNPlugind.so")
            set(libmyriadPluginPath "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/libmyriadPlugind.so")
        else()
            set(libMKLDNNPluginPath "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/libMKLDNNPlugin.so")
            set(libmyriadPluginPath "${CMAKE_CURRENT_BINARY_DIR}/python/openvino_tensorflow/libmyriadPlugin.so")
        endif()

        # libMKLDNNPluginPath
        execute_process(COMMAND
            install_name_tool -add_rpath
            @loader_path
            ${libMKLDNNPluginPath}
            RESULT_VARIABLE result
            ERROR_VARIABLE ERR
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(${result})
             message(FATAL_ERROR "Cannot add rpath")
        endif()

        # libmyriadPluginPath
        execute_process(COMMAND
            install_name_tool -add_rpath
            @loader_path
            ${libmyriadPluginPath}
            RESULT_VARIABLE result
            ERROR_VARIABLE ERR
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(${result})
             message(FATAL_ERROR "Cannot add rpath")
        endif()
     endif()

    execute_process(
        COMMAND ${PYTHON} "setup.py" "bdist_wheel"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/
    )

endif()
