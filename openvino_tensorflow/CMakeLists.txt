 # Copyright (C) 2021-2022 Intel Corporation

 # SPDX-License-Identifier: Apache-2.0

set(LIB_NAME openvino_tensorflow)

include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)
include_directories(${NGRAPH_INSTALL_DIR}/include)
include_directories(${ABSL_INCLUDE_DIR})

set(TF_CONVERSION_EXTENSIONS_MODULE_NAME "tf_conversion_extensions")
add_compile_definitions(TF_CONVERSION_EXTENSIONS_MODULE_NAME="${TF_CONVERSION_EXTENSIONS_MODULE_NAME}")
add_subdirectory(tf_conversion_extensions)

# For some reason the following is needed for ABI compatibility with TF.
# (There must be some dependency on it in the struct/class definitions.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
set(SRC
   api.cc
   backend.cc
   backend_manager.cc
   executable.cc
   ie_tensor.cc
   kernels/encapsulate_op.cc
   assign_clusters.cc
   ovtf_builder.cc
   cluster_manager.cc
   layout_conversions.cc
   deassign_clusters.cc
   encapsulate_clusters.cc
   mark_for_clustering.cc
   rewrite_pass.cc
   ovtf_utils.cc
   ops/encapsulate_op.cc
   pass/transpose_sinking.cc
   tf_graphcycles.cc
   tf_deadness_analysis.cc
   version.cc
   ie_backend_engine.cc
   ie_basic_engine.cc
   ie_vadm_engine.cc
   grappler/ovtf_optimizer.cc
   grappler/add_identityn.cc
   grappler/costs/measuring_cost_estimator.cc
   grappler/costs/robust_stats.cc
   grappler/costs/cost_analyzer.cc
   ovtf_decoder.cc
)

string(REPLACE "." ";" TF_VERSION_LIST ${TensorFlow_VERSION})
list(GET TF_VERSION_LIST 0 TF_MAJOR_VERSION)
list(GET TF_VERSION_LIST 1 TF_MINOR_VERSION)

# Enable Grappler only for TF >= 2.8
if(NOT ((TF_MAJOR_VERSION STRGREATER_EQUAL "2") AND (TF_MINOR_VERSION STRGREATER_EQUAL "8")))
   list(REMOVE_ITEM SRC grappler/ovtf_optimizer.cc)
   list(REMOVE_ITEM SRC grappler/add_identityn.cc)
   list(REMOVE_ITEM SRC grappler/costs/measuring_cost_estimator.cc)
   list(REMOVE_ITEM SRC grappler/costs/robust_stats.cc)
   list(REMOVE_ITEM SRC grappler/costs/cost_analyzer.cc)
endif()

if(OPENVINO_BUILD_VERSION)
    add_compile_definitions(OPENVINO_BUILD_VERSION="${OPENVINO_BUILD_VERSION}")
endif()

add_library(${LIB_NAME} SHARED ${SRC})

target_link_libraries(
        ${LIB_NAME}
        ovtf_logger
        ${TensorFlow_FRAMEWORK_LIBRARY}
        absl_algorithm
        absl_container
        absl_strings
        ${InferenceEngine_LIBRARIES}
        tensorflow_frontend_lib
        ocm
        tf_conversion_extensions
    )

target_compile_definitions(
    ${LIB_NAME} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}
)

target_include_directories(${LIB_NAME} PUBLIC "${NGRAPH_INSTALL_DIR}/include")

add_compile_definitions(TF_VERSION=${TF_MAJOR_VERSION})

#------------------------------------------------------------------------------
#installation
#------------------------------------------------------------------------------
if (DEFINED OPENVINO_TF_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${OPENVINO_TF_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install/")
endif()

if(OS_VERSION STREQUAL "\"centos\"")
	set(OVTF_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib64)
else()
	set(OVTF_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()

# OVTF install dir required for importing TF Conversion Extensions library
add_compile_definitions(OVTF_INSTALL_LIB_DIR="${OVTF_INSTALL_LIB_DIR}")

message(STATUS "OVTF_INSTALL_LIB_DIR: ${OVTF_INSTALL_LIB_DIR}")

# Install the openvino_tensorflow lib
install(TARGETS ${LIB_NAME} DESTINATION "${OVTF_INSTALL_LIB_DIR}")

string(REPLACE "." ";" OPENVINO_VERSION_LIST ${OPENVINO_VERSION})
list(GET OPENVINO_VERSION_LIST 0 OPENVINO_VERSION_MAJOR)
string(SUBSTRING ${OPENVINO_VERSION_MAJOR} 2 2 OPENVINO_VERSION_MAJOR)
list(GET OPENVINO_VERSION_LIST 1 OPENVINO_VERSION_MINOR)
list(GET OPENVINO_VERSION_LIST 2 OPENVINO_VERSION_PATCH)
set(OPENVINO_SO_VER "${OPENVINO_VERSION_MAJOR}${OPENVINO_VERSION_MINOR}${OPENVINO_VERSION_PATCH}")

# Install OCM and set a variable with required OpenVINO libraries path,
if (APPLE)
  set(OCM_LIB libocm.dylib)
  set(TF_CONVERSION_EXTENSIONS_LIB "lib${TF_CONVERSION_EXTENSIONS_MODULE_NAME}.dylib")
  set(LIB_PREFIX "lib")
  set(PLUGIN_LIB_EXT "so")
  
  if ((OPENVINO_SO_VER STRGREATER_EQUAL "2230"))
    set(OV_LIB_EXT_DOT "${OPENVINO_VERSION}.dylib")
    set(OV_LIB_EXT "${OPENVINO_SO_VER}.dylib")
  else()
    set(OV_LIB_EXT ".dylib")
  endif()

  set(IE_LIBS_PATH ${OPENVINO_ARTIFACTS_DIR}/runtime/lib/intel64/Release)
  set(IE_LIBS 
      "${IE_LIBS_PATH}/pcie-ma2x8x.mvcmd" 
      )
  set(TBB_LIBS ${OPENVINO_ARTIFACTS_DIR}/runtime/3rdparty/tbb/lib/)
  install(FILES ${CMAKE_INSTALL_PREFIX}/../ocm/OCM/${OCM_LIB} DESTINATION "${OVTF_INSTALL_LIB_DIR}")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${TF_CONVERSION_EXTENSIONS_MODULE_NAME}/${TF_CONVERSION_EXTENSIONS_LIB} DESTINATION "${OVTF_INSTALL_LIB_DIR}")
elseif(WIN32)
  set(OCM_LIB ocm.dll)
  set(TF_CONVERSION_EXTENSIONS_LIB "${TF_CONVERSION_EXTENSIONS_MODULE_NAME}.dll")
  set(LIB_PREFIX "")
  set(PLUGIN_LIB_EXT "dll")
  SET(OV_LIB_EXT ${PLUGIN_LIB_EXT})
  set(IE_LIBS_PATH ${OPENVINO_ARTIFACTS_DIR}/runtime/bin/intel64/Release)
  set (IE_LIBS 
      "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_intel_gpu_plugin.${PLUGIN_LIB_EXT}"
      "${IE_LIBS_PATH}/cache.json"
      "${IE_LIBS_PATH}/pcie-ma2x8x.elf"
     )
  set(TBB_LIBS ${OPENVINO_ARTIFACTS_DIR}/runtime/3rdparty/tbb/bin/)
  install(FILES ${CMAKE_INSTALL_PREFIX}/../ocm/OCM/Release/${OCM_LIB} DESTINATION "${OVTF_INSTALL_LIB_DIR}")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${TF_CONVERSION_EXTENSIONS_MODULE_NAME}/Release/${TF_CONVERSION_EXTENSIONS_LIB} DESTINATION "${OVTF_INSTALL_LIB_DIR}")
else()
  set(OCM_LIB libocm.so)
  set(TF_CONVERSION_EXTENSIONS_LIB "lib${TF_CONVERSION_EXTENSIONS_MODULE_NAME}.so")
  set(LIB_PREFIX "lib")
  set(PLUGIN_LIB_EXT "so")

  if ((OPENVINO_SO_VER STRGREATER_EQUAL "2230"))
    set(OV_LIB_EXT_DOT "${PLUGIN_LIB_EXT}.${OPENVINO_VERSION}")
    set(OV_LIB_EXT "${PLUGIN_LIB_EXT}.${OPENVINO_SO_VER}")
  else()
    set(OV_LIB_EXT "${PLUGIN_LIB_EXT}")
  endif()

  set(IE_LIBS_PATH ${OPENVINO_ARTIFACTS_DIR}/runtime/lib/intel64)
  set (IE_LIBS 
      "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_intel_gpu_plugin.${PLUGIN_LIB_EXT}"
      "${IE_LIBS_PATH}/cache.json"
      "${IE_LIBS_PATH}/pcie-ma2x8x.mvcmd" 
     )
  set(TBB_LIBS ${OPENVINO_ARTIFACTS_DIR}/runtime/3rdparty/tbb/lib/)
  install(FILES ${CMAKE_INSTALL_PREFIX}/../ocm/OCM/${OCM_LIB} DESTINATION "${OVTF_INSTALL_LIB_DIR}")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${TF_CONVERSION_EXTENSIONS_MODULE_NAME}/${TF_CONVERSION_EXTENSIONS_LIB} DESTINATION "${OVTF_INSTALL_LIB_DIR}")
endif()

# Update this list for any new OpenVINO component/feature addition in future
set (IE_LIBS 
    "${IE_LIBS}" 
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino.${OV_LIB_EXT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_c.${OV_LIB_EXT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_tensorflow_fe.${OV_LIB_EXT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino.${OV_LIB_EXT_DOT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_c.${OV_LIB_EXT_DOT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_tensorflow_fe.${OV_LIB_EXT_DOT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_intel_cpu_plugin.${PLUGIN_LIB_EXT}"
    "${IE_LIBS_PATH}/${LIB_PREFIX}openvino_intel_myriad_plugin.${PLUGIN_LIB_EXT}"
    "${IE_LIBS_PATH}/usb-ma2x8x.mvcmd" 
    "${IE_LIBS_PATH}/plugins.xml" 
    )

# Install Openvino and TBB libraries
if(NOT DISABLE_PACKAGING_OPENVINO_LIBS)
  file(GLOB TBB_LIB_FILES "${TBB_LIBS}/*")
  set(OPENVINO_LIB_FILES "${IE_LIBS};${TBB_LIB_FILES}")

  foreach(LIB_FILE ${OPENVINO_LIB_FILES})
      file(COPY ${LIB_FILE} DESTINATION ${OVTF_INSTALL_LIB_DIR})
      get_filename_component(lib_file_name ${LIB_FILE} NAME)
      set(lib_file_real_path "${OVTF_INSTALL_LIB_DIR}/${lib_file_name}")
      if(${lib_file_name} MATCHES ".so*")
          install(CODE "
          execute_process(COMMAND patchelf --set-rpath $ORIGIN ${lib_file_real_path} COMMAND_ECHO STDOUT)
          ")
      endif()
  endforeach()
endif()

# Now create the Python pip package. The following variables are passed to the
# CreatePipWhl.cmake
# CMake "install" target "CODE" will propagate these variables to the
# target "SCRIPT" so that the "SCRIPT" (in ths case CreatePipWhl.cmake)
# So any variable that we need to propagate needs to be added here
install(CODE "set(OS_VERSION ${OS_VERSION})")
install(CODE "set(NGRAPH_INSTALL_DIR \"${NGRAPH_INSTALL_DIR}\")")
install(CODE "set(OVTF_SRC_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/../\")")
install(CODE "set(OVTF_INSTALL_DIR \"${CMAKE_INSTALL_PREFIX}\")")
install(CODE "set(TensorFlow_GIT_VERSION \"${TensorFlow_GIT_VERSION}\")")
install(CODE "set(TensorFlow_VERSION \"${TensorFlow_VERSION}\")")
install(CODE "set(CMAKE_BUILD_TYPE \"${CMAKE_BUILD_TYPE}\")")
if(PYTHON_EXECUTABLE)
    install(CODE "set(PYTHON_EXECUTABLE \"${PYTHON_EXECUTABLE}\")")
endif()
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../python/CreatePipWhl.cmake")
