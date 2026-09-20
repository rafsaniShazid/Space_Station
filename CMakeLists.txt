cmake_minimum_required(VERSION 3.20)
project(space_station LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(space_station src/main.cpp)
target_link_libraries(space_station PRIVATE OpenGL::GL GLUT::GLUT)

if(WIN32)
	target_link_libraries(space_station PRIVATE glu32)
else()
	target_link_libraries(space_station PRIVATE OpenGL::GLU)
endif()

target_compile_options(space_station PRIVATE -Wall -Wextra -Wpedantic)
