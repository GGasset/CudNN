#ifndef CUDA_ACTIVATIONS
#define CUDA_ACTIVATIONS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "neuron_operations.cuh"

#include <cmath>


__global__ void sigmoid_activation(
	data_t *activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t *execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values
)
{
	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * threadIdx.x;
	data_t activation = 1 / (1 + exp(-execution_values[neuron_execution_values_start + neuron_execution_values_read]));
	if (write_activation)
	{
		size_t activations_i = activations_start + layer_activation_start + threadIdx.x;
		activations[activations_i] = activation;
	}
	if (write_execution_values)
	{
		size_t execution_values_i = execution_values_start + execution_values_layer_start + execution_values_per_neuron * threadIdx.x + neuron_execution_values_write;
		execution_values[execution_values_i] = activation;
	}
}

__global__ void tanh_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values
)
{
	
}

#endif