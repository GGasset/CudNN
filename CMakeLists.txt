cmake_minimum_required(VERSION 3.28)

# Add linking to stdlib
SET(GCC_COVERAGE_COMPILE_FLAGS "-lstdc++")
SET(GCC_COVERAGE_LINK_FLAGS    "-lstdc++")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

# -rdc=true
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin='g++-13'")

project(NN LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}")


add_executable(NN
data_type.h
DenseConnections.h
evolution_info.h
functionality.h
IConnections.h
ILayer.h
kernel_macros.h
LSTMLayer.h
NeatConnections.h
NeuronLayer.h
NN.h

connection_gradients.cu
costs.cu
cuda_functionality.cu
DenseConnections.cu
derivatives.cu
gradients.cu
IConnections.cu
ILayer.cu
kernel.cu
linear_functions.cu
LSTMLayer.cu
NeatConnections.cu
NeuronLayer.cu
neuron_operations.cu
NN.cu

functionality.cpp
)


target_link_libraries(NN -lcurand)
set_property(TARGET NN PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(NN PROPERTIES CUDA_ARCHITECTURES "native")

