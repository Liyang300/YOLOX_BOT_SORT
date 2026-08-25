set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
cmake_minimum_required(VERSION 3.14)
project(YoloPreprocessDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
# Eigen3
find_package(Eigen3 REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "/opt/onnxruntime")

# 源码收集
file(GLOB SRC_FILES 
    ${PROJECT_SOURCE_DIR}/source/*.cpp
    ${PROJECT_SOURCE_DIR}/botsort/src/*.cpp
)

# ========== 创建可执行目标 yolo_pre ==========
add_executable(yolo_pre ${SRC_FILES})

#========target_xxx放到add_executable之后 ==========
target_include_directories(yolo_pre PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/botsort/inc
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

# 链接库
target_link_libraries(yolo_pre
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)