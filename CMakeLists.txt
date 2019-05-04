cmake_minimum_required(VERSION 3.14)
project(count_people)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(count_people main.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
target_link_libraries(count_people ${OpenCV_LIBS})