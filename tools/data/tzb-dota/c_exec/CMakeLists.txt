cmake_minimum_required(VERSION 3.16)
project(dota_img_split)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall --std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})

find_package(GDAL REQUIRED)

include_directories(${PRoject_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(${CMAKE_PROJECT_NAME} ${DIR_SRCS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXTRA_LIBS ${EXTRA_LIBS} GDAL::GDAL)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${EXTRA_LIBS})

add_definitions(-O0 -pthread)


