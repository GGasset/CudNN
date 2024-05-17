#ifndef CUDA_LINEAR_FUNCTIONS
#define CUDA_LINEAR_FUNCTIONS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "linear_functions.cuh"

#include "data_type.h"

__global__ void cud_dense_linear_function(
	size_t previous_layer_length, field_t* weights,
	size_t activations_start, size_t previous_layer_activations_start, data_t* activations,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{
	size_t connected_activation_i = activations_start + previous_layer_activations_start + threadIdx.x;
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * blockIdx.x;

	field_t current_weight = weights[previous_layer_length * blockIdx.x + threadIdx.x];
	atomicAdd(execution_values + execution_values_i, current_weight * activations[connected_activation_i]);
	//execution_values[execution_values_i] += current_weight * activations[connected_activation_i];
}

__global__ void cud_add_biases(
	field_t* biases,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * threadIdx.x;
	execution_values[execution_values_i] += biases[threadIdx.x];
}

__global__ void cud_dense_linear_function_derivative(
	size_t activations_start, size_t previous_layer_activations_start, size_t previous_layer_length, data_t* activations,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
	field_t* weights
)
{
	size_t activation_i = activations_start + previous_layer_activations_start + threadIdx.x;
	size_t weight_i = previous_layer_length * blockIdx.x + threadIdx.x;
	size_t connection_derivative = activations[activation_i] + weights[weight_i];

	size_t write_i = derivatives_start + derivatives_layer_start + derivatives_per_neuron * blockIdx.x;
	derivatives[write_i] += connection_derivative;
}

__global__ void cud_add_bias_derivative(
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
)
{
	derivatives[derivatives_start + derivatives_layer_start + derivatives_per_neuron * threadIdx.x] += 1;
}

#endif