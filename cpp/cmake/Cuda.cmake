# Known NVIDIA GPU achitectures Annfab can be compiled for.
# This list will be used for CUDA_ARCH_NAME = All option
set(Annfab_known_gpu_archs "20 21(20) 30 35 50")

################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   annfab_select_nvcc_arch_flags(out_variable)
function(annfab_select_nvcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Fermi" "Kepler" "Maxwell" "All" "Manual")
  set(__archs_name_default "All")

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architecture names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN ${Annfab_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(CUDA_ARCH_PTX "50"                      CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Fermi")
    set(__cuda_arch_bin "20 21(20)")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${Annfab_known_gpu_archs})
  else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(__cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
  # Ensure that there are no duplicates in the lists:
  list(REMOVE_DUPLICATES __cuda_arch_bin)
  list(REMOVE_DUPLICATES __cuda_arch_ptx)

  set(__nvcc_flags "")
  set(__nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(__arch ${__cuda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __nvcc_archs_readable sm_${__arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__cuda_arch_ptx})
    list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __nvcc_archs_readable compute_${__arch})
  endforeach()

  string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for cuda comnpilation
# Usage:
#   annfab_cuda_compile(<objlist_variable> <cuda_files>)
macro(annfab_cuda_compile objlist_variable)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var}_backup_in_cuda_compile_ "${${var}}")

    # we remove /EHa as it generates warnings under windows
    string(REPLACE "/EHa" "" ${var} "${${var}}")

  endforeach()

  if(UNIX OR APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC, -std=c++11)
  endif()

  if(APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -Wno-unused-function)
  endif()
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  cuda_compile(cuda_objcs ${ARGN})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_cuda_compile_}")
    unset(${var}_backup_in_cuda_compile_)
  endforeach()

  set(${objlist_variable} ${cuda_objcs})
endmacro()


################################################################################################
###  Non macro section
################################################################################################

find_package(CUDA 5.5 QUIET)
find_cuda_helper_libs(curand)  # cmake 2.8.7 compartibility which doesn't search for curand

if(NOT CUDA_FOUND)
  return()
endif()

set(HAVE_CUDA TRUE)
message(STATUS "CUDA detected: " ${CUDA_VERSION})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND annfab_LINKER_LIBS ${CUDA_CUDART_LIBRARY}
                              ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

# setting nvcc arch flags
annfab_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")

# Boost 1.55 workaround, see https://svn.boost.org/trac/boost/ticket/9392 or
# https://github.com/ComputationalRadiationPhysics/picongpu/blob/master/src/picongpu/CMakeLists.txt
if(Boost_VERSION EQUAL 105500)
  message(STATUS "Cuda + Boost 1.55: Applying noinline work around")
  # avoid warning for CMake >= 2.8.12
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif()

# disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  list(APPEND CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# setting default testing device
if(NOT CUDA_TEST_DEVICE)
  set(CUDA_TEST_DEVICE -1)
endif()

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)
