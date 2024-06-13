cmake_minimum_required(VERSION 3.28)


# -rdc=true
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin='gcc-13'")

project(NN LANGUAGES CUDA CXX)

add_executable(NN kernel.cu)

set_property(TARGET NN PROPERTY CUDA_SEPARABLE_COMPILATION ON)
