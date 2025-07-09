cmake_minimum_required(VERSION 3.10)
project(YOLO11_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

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

# Add source files
set(SOURCES
    src/ia/tools/Config.cpp
    src/ia/YOLO11.cpp
    src/ia/YOLOSeg.cpp
)

# Create library
add_library(yolo11_lib STATIC ${SOURCES})

# Link libraries to the static library
target_link_libraries(yolo11_lib ${OpenCV_LIBS})
if(ONNXRUNTIME_LIB)
    target_link_libraries(yolo11_lib ${ONNXRUNTIME_LIB})
else()
    message(WARNING "ONNX Runtime not found. Please install ONNX Runtime.")
endif()

# Main executable
add_executable(yolov11_detector src/main.cpp)
target_link_libraries(yolov11_detector yolo11_lib)

# Model info executable
add_executable(model_info model_info.cpp)
target_link_libraries(model_info yolo11_lib)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(yolo11_lib PRIVATE -Wall -Wextra -O3)
    target_compile_options(yolov11_detector PRIVATE -Wall -Wextra -O3)
    target_compile_options(model_info PRIVATE -Wall -Wextra -O3)
endif()

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
else()
    message(STATUS "ONNX Runtime not found")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Copy model files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/chpt DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/classes.txt DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/best.onnx DESTINATION ${CMAKE_BINARY_DIR}) 