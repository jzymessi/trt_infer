cmake_minimum_required(VERSION 3.16)

project(TRT_Infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_Release "-O2 -g")


find_package(CUDAToolkit)
include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(main main.cpp)

target_link_libraries(main 
        ${OpenCV_LIBS}
        ${OpenMP_CXX_LIBRARIES}
        resnet
        fpn
        yolo
        utils
        nvinfer
        cudart
        )