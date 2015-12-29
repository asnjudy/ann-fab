cmake_minimum_required(VERSION 2.8.7)

# ---[ annfab project
project(annfab C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ Options
option(CPU_ONLY  "Build ann-fab without CUDA support" OFF)

# ---[ Flags
if(UNIX OR APLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wno-sign-compare")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(annfab_DEBUG_POSTFIX "-d")
set(annfab_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# ---[ Includes
include(cmake/Targets.cmake)
include(cmake/Dependencies.cmake)
include_directories(${annfab_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
include_directories ("${annfab_INCLUDE_DIR}/src")

# add the source file subdirectory (and run its cmake file)
add_subdirectory(src/annfab)
add_subdirectory(python)
