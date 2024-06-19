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
	size_t tid = get_tid();
	if (tid >= previous_layer_length)
		return;
	size_t connected_activation_i = activations_start + previous_layer_activations_start + tid;
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * blockIdx.y;

	field_t current_weight = weights[previous_layer_length * blockIdx.y + tid];
	atomicAdd(execution_values + execution_values_i, current_weight * activations[connected_activation_i]);
}

__global__ void cud_NEAT_neuron_linear_function(
	size_t neuron_i, size_t connection_count, field_t* weights, size_t* connection_points, size_t connections_start,
	size_t activations_start, data_t* activations, 
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values
)
{
	size_t tid = get_tid();
	if (tid >= connection_count)
		return;

	size_t connection_i = connection_points[connections_start + tid];
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * neuron_i;
	atomicAdd(execution_values + execution_values_i, activations[activations_start + connection_i] * weights[connections_start + tid]);
}

__global__ void cud_add_biases(
	size_t layer_length, field_t* biases,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values)
{
	size_t tid = get_tid();
	if (tid >= layer_length)
		return;
	size_t execution_values_i = execution_values_start + execution_values_layer_start + layer_execution_values_per_neuron * tid;
	atomicAdd(execution_values + execution_values_i, biases[tid]);
}

__global__ void cud_dense_linear_function_derivative(
	size_t activations_start, size_t previous_layer_activations_start, size_t previous_layer_length, data_t* activations,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
	field_t* weights
)
{
	size_t tid = get_tid();
	if (tid >= previous_layer_length)
		return;
	size_t activation_i = activations_start + previous_layer_activations_start + tid;
	size_t weight_i = previous_layer_length * blockIdx.y + tid;
	size_t connection_derivative = activations[activation_i] + weights[weight_i];

	size_t write_i = derivatives_start + derivatives_layer_start + derivatives_per_neuron * blockIdx.y;
	atomicAdd(derivatives + write_i, connection_derivative);
}

__global__ void cud_NEAT_linear_function_derivative(
	size_t activations_start, data_t* activations,
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
	size_t neuron_i, size_t connection_count, field_t* weights, size_t* connection_points, size_t connections_start
)
{
	size_t tid = get_tid();
	if (tid >= connection_count)
		return;

	size_t activation_i = connection_points[connections_start + tid];
	size_t weight_i = connections_start + tid;
	size_t write_i = derivatives_start + derivatives_layer_start + derivatives_per_neuron * neuron_i;
	atomicAdd(derivatives + write_i, activations[activation_i] + weights[weight_i]);
}

__global__ void cud_add_bias_derivative(
	size_t layer_length, 
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
)
{
	size_t tid = get_tid();
	if (tid >= layer_length)
		return;
	atomicAdd(derivatives + derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid, 1);
}

#endif
