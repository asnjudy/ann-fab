set(annfab_LINKER_LIBS "")

# ---[ CUDA
if(CPU_ONLY)
  message("-- CUDA is disabled. Building without it...")
  add_definitions(-DCPU_ONLY)
else()
  include(cmake/Cuda.cmake)
  if(NOT HAVE_CUDA)
    message("-- CUDA not detected by cmake. Building without it...")
    add_definitions(-DCPU_ONLY)
  endif()
endif()


# ---[ Python
find_package(PythonInterp 2.7)
find_package(PythonLibs 2.7)
find_package(NumPy 1.7.1)
find_package(Boost 1.46 COMPONENTS python)

if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
  set(HAVE_PYTHON TRUE)
  include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
  list(APPEND annfab_LINKER_LIBS ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
endif()

# ---[ BLAS
find_package(OpenBLAS REQUIRED)
include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
list(APPEND annfab_LINKER_LIBS ${OpenBLAS_LIB})
