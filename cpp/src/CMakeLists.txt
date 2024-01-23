include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_library(resnet SHARED resnet.cpp)
set_target_properties(resnet PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib
)

add_library(fpn SHARED fpn.cpp)
set_target_properties(fpn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib
)

add_library(yolox SHARED yolox.cpp)
set_target_properties(yolox PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib
)

add_library(utils SHARED utils.cpp)
set_target_properties(utils PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib
)

