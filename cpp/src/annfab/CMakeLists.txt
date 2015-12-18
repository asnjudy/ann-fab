# add all the .cpp files to the src list
file(GLOB annfab_SRC "*.cpp")
file(GLOB cuda "*.cu")
include_directories(${annfab_SOURCE_DIR}/include)

annfab_cuda_compile(cuda_objs ${cuda})
list(APPEND annfab_SRC ${cuda_objs} ${cuda})

add_library(annfab ${annfab_SRC})
annfab_default_properties(annfab)
target_link_libraries(annfab ${annfab_LINKER_LIBS})
