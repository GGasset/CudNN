#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

#pragma once

__global__ void cud_dense_linear_function(
	size_t previous_layer_length, parameter_t* weights,
	size_t activations_start, size_t previous_layer_activations_start, data_t* activations,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{
	size_t connected_activation_i = activations_start + previous_layer_activations_start + threadIdx.x;
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * blockIdx.x;

	parameter_t current_weight = weights[previous_layer_length * blockIdx.x + threadIdx.x];
	execution_values[execution_values_i] += current_weight * activations[connected_activation_i];
}

__global__ void cud_add_biases(
	parameter_t* biases,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * threadIdx.x;
	execution_values[execution_values_i] += biases[threadIdx.x];
}