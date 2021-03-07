 # Copyright (C) 2021 Intel Corporation
 
 # SPDX-License-Identifier: Apache-2.0

if(NOT EXISTS "@CMAKE_BINARY_DIR@/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: @CMAKE_BINARY_DIR@/install_manifest.txt")
endif(NOT EXISTS "@CMAKE_BINARY_DIR@/install_manifest.txt")

file(READ "@CMAKE_BINARY_DIR@/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  get_filename_component( DIR ${file} DIRECTORY )
  list(APPEND DoomedDirList ${DIR})
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    file( REMOVE_RECURSE ${file})  
  else()
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif()
endforeach(file)

# Now remove the directories
list(REMOVE_DUPLICATES DoomedDirList)
foreach( dir ${DoomedDirList})
  message( STATUS "Removing directory: " ${dir} )
  file(REMOVE_RECURSE ${dir})
endforeach()
