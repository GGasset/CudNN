#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

#pragma once

__device__ void cud_dense_partial_linear_function(
	size_t* weights_starts, parameter_t* weights,
	size_t previous_layer_activations_start, data_t* activations,
	size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{
	size_t connected_activation_i = previous_layer_activations_start + threadIdx.x;
	size_t execution_values_i = execution_values_layer_start + layer_execution_values_per_neuron * blockIdx.x;

	parameter_t current_weight = weights[weights_starts[blockIdx.x] + threadIdx.x];
	execution_values[execution_values_i] += current_weight * activations[connected_activation_i];
}

__global__ void cud_dense_linear_function(
	size_t* weights_starts, parameter_t* weights, parameter_t* biases,
	size_t previous_layer_activations_start, data_t* activations,
	size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{

}