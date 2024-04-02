#pragma once

#include "data_type.h"

__global__ void cud_add_biases(
	parameter_t* biases,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values);

__global__ void cud_dense_linear_function(
	size_t previous_layer_length, parameter_t* weights,
	size_t activations_start, size_t previous_layer_activations_start, data_t* activations,
	size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, data_t* execution_values);