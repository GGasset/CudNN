cmake_minimum_required(VERSION 3.28)

# Add linking to stdlib
SET(GCC_COVERAGE_COMPILE_FLAGS "-lm -lstdc++ -fanalyzer")
SET(GCC_COVERAGE_LINK_FLAGS    "-lm -lstdc++")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

set (CMAKE_CXX_STANDARD 20)

# -rdc=true
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin='g++-13'")
if (DEBUG)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
	add_compile_definitions("DEBUG")
	message("-- Config debug")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native")

project(NN LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}")
if(DETERMINISTIC)
	add_compile_definitions(DETERMINISTIC)
	message("-- Deterministic value generation active")
endif()
add_compile_definitions(INCLUDE_BACKEND)
if(target STREQUAL "static")
	   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")
	   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")
	   message("-- Static Lib added")
	   add_library(${PROJECT_NAME} STATIC	
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
		NN_constructor.h

		connection_gradients.cu
		costs.cu
		cuda_functionality.cu
		DenseConnections.cu
		derivatives.cu
		gradients.cu
		IConnections.cu
		ILayer.cu
		linear_functions.cu
		LSTMLayer.cu
		NeatConnections.cu
		NeuronLayer.cu
		neuron_operations.cu
		NN.cu
		NN_constructor.cpp

		NN_enums.h

		functionality.cpp
	)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
		   "Built Static Lib")
elseif(target STREQUAL "socket")
	message("-- Socket config added")
	add_executable(${PROJECT_NAME}
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
		NN_constructor.h

		connection_gradients.cu
		costs.cu
		cuda_functionality.cu
		DenseConnections.cu
		derivatives.cu
		gradients.cu
		IConnections.cu
		ILayer.cu
		linear_functions.cu
		LSTMLayer.cu
		NeatConnections.cu
		NeuronLayer.cu
		neuron_operations.cu
		NN.cu
		NN_constructor.cpp

		functionality.cpp

		NN_socket_interpreter.h
		NN_socket_interpreter.cpp

		unix_sock_interop.h
		unix_sock_interop.cpp

		NN_enums.h

		HashTable.h
		SinglyLinkedListNode.h
	)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
	   "Built socket config")
else()
	message("-- Exe for testing added")
	add_executable(${PROJECT_NAME}
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
		NN_constructor.h

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
		NN_constructor.cpp

		NN_enums.h

		functionality.cpp
	)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC -lcurand)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "native")
if (DEBUG)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
                   "Built debug")
endif()
