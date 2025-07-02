cmake_minimum_required(VERSION 3.12)
project(YOLOv11_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Try to find ONNX Runtime
set(ONNXRUNTIME_ROOT_PATH /usr/local)
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
          ${ONNXRUNTIME_ROOT_PATH}/include
          /usr/include/onnxruntime
          /usr/local/include/onnxruntime
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT_PATH}/lib
          /usr/lib
          /usr/local/lib
)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(src)

# Add executable
add_executable(yolov11_detector
    src/main.cpp
    src/ia/YOLO11.cpp
    src/ia/tools/Config.cpp
)

# Link libraries
target_link_libraries(yolov11_detector ${OpenCV_LIBS})
if(ONNXRUNTIME_LIB)
    target_link_libraries(yolov11_detector ${ONNXRUNTIME_LIB})
else()
    message(WARNING "ONNX Runtime not found. Please install ONNX Runtime.")
endif()

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(yolov11_detector PRIVATE -Wall -Wextra -O3)
endif()

# Copy model and classes files to build directory
configure_file(${CMAKE_SOURCE_DIR}/best.onnx ${CMAKE_BINARY_DIR}/best.onnx COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/classes.txt ${CMAKE_BINARY_DIR}/classes.txt COPYONLY) 