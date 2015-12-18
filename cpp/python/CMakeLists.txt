
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
file(GLOB_RECURSE python_srcs ${PROJECT_SOURCE_DIR}/python/*.cpp)

add_library(pyannfab SHARED ${python_srcs})
target_link_libraries(pyannfab annfab ${annfab_LINKER_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(pyannfab PROPERTIES PREFIX "" OUTPUT_NAME "_annfab")
annfab_default_properties(pyannfab)