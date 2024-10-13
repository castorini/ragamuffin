# Build ragamuffin as a shared library
file(GLOB_RECURSE LIB_SOURCES "lib/*.cpp")
find_package(CUDAToolkit REQUIRED)

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS})
add_library(ragamuffin STATIC ${LIB_SOURCES})
target_link_libraries(ragamuffin ${CUDAToolkit_LIBRARIES})

# Iterate over all .cu files in samples directory and build a target for each
file(GLOB_RECURSE SAMPLES "samples/*.cu")

foreach(file IN LISTS SAMPLES)
    get_filename_component(TARGET_NAME ${file} NAME)
    include_directories(${CMAKE_SOURCE_DIR}/lib)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_executable(${TARGET_NAME} ${file})
    target_link_libraries(${TARGET_NAME} ragamuffin ${CUDAToolkit_LIBRARIES})
endforeach()
