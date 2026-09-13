cmake_minimum_required(VERSION 3.15)
project(ONNX_Camera CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime - Local prebuilt GPU binaries (CUDA12 build, ORT 1.18.0).
# Self-contained copy for this project.
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-gpu-1.18.0")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR})
    message(FATAL_ERROR "ONNX Runtime include directory not found: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${ONNXRUNTIME_LIBRARY})
    message(FATAL_ERROR "ONNX Runtime library not found: ${ONNXRUNTIME_LIBRARY}")
endif()

message(STATUS "Using ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

add_executable(onnx_camera
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)
target_link_libraries(onnx_camera
    PRIVATE
        ${ONNXRUNTIME_LIBRARY}
        ${OpenCV_LIBS}
)
set_target_properties(onnx_camera PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)
