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

set(MY_COMMON_INC
    ${PROJECT_SOURCE_DIR}/common/include
)

# 源码收集，公共资源
file(GLOB COMMON_SRC 
    ${PROJECT_SOURCE_DIR}/common/source/*.cpp
)

# ---------------------- Bytetrack 可执行文件 ----------------------
# bytetrack专属源码：bytetrack src + bytetrack自己的main
file(GLOB BYTETRACK_SRC
    ${PROJECT_SOURCE_DIR}/test/bytetrack_test.cpp   # bytetrack的main
    ${PROJECT_SOURCE_DIR}/TrackMod/bytetrack/src/*.cpp
    ${COMMON_SRC}
)

# ========== 创建可执行目标 ==========
add_executable(app_bytetrack ${BYTETRACK_SRC})

#========target_xxx放到add_executable之后 ==========
target_include_directories(app_bytetrack PRIVATE
    ${PROJECT_SOURCE_DIR}/TrackMod/bytetrack/inc
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    ${MY_COMMON_INC}
)

# 链接库
target_link_libraries(app_bytetrack
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# ---------------------- Botsort 可执行文件 ----------------------
# botsort专属源码：botsort src + botsort自己的main
file(GLOB BOTSORT_SRC
    ${PROJECT_SOURCE_DIR}/test/botsort_test.cpp   # botsort的main
    ${PROJECT_SOURCE_DIR}/TrackMod/botsort/src/*.cpp
    ${COMMON_SRC}
)

add_executable(app_botsort ${BOTSORT_SRC})

target_include_directories(app_botsort PRIVATE
    ${PROJECT_SOURCE_DIR}/TrackMod/botsort/inc
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    ${MY_COMMON_INC}
)

# 链接库
target_link_libraries(app_botsort
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)
