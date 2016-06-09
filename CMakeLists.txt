cmake_minimum_required(VERSION 2.8.4)

project(marsupial)

set(CMAKE_BUILD_TYPE Release)

add_library(dlib SHARED IMPORTED)
set_target_properties(dlib PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/dlib/build/libdlib.so)
include(dlib/cmake)


option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions" OFF)

IF(USE_AVX_INSTRUCTIONS)
  add_definitions(-mavx)
  add_definitions(-march=native)
ENDIF()

include_directories(${CMAKE_JS_INC})

file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} dlib)

