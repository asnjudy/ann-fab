set(annfab_LINKER_LIBS "")

# ---[ CUDA
include(cmake/Cuda.cmake)

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
