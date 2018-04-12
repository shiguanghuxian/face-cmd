#1.cmake verson，指定cmake版本 
cmake_minimum_required(VERSION 2.6)

#2.project name，指定项目的名称
PROJECT(face)

## linux 根据实际情况设置
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(QMAKE_CXXFLAGS "-std=c++11")

#3.head file path，头文件目录
include_directories(${seeta_facedet_lib_INCLUDE_DIRS} ${seeta_fa_lib_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(
    ../FaceIdentification/include
    ../FaceAlignment/include
    ../FaceDetection/include
)

#4. add external libraries 添加外部库
find_package(OpenCV REQUIRED)

list(APPEND seeta_fi_lib_required_libs ${OpenCV_LIBS} seeta_facedet_lib seeta_fa_lib)

#5.source directory，源文件目录
AUX_SOURCE_DIRECTORY(src SRC_LIST)
link_directories("${PROJECT_SOURCE_DIR}/libs")
# message(${SRC_LIST})

#6.add executable file，添加要编译的可执行文件
ADD_EXECUTABLE(face ./src/main.cpp)

#7.add link library，添加可执行文件所需要的库
target_link_libraries(face viplnet ${seeta_fi_lib_required_libs})
