#pragma once

#include "data_type.h"

enum ActivationFunctions
{
	sigmoid,
	tanh
};

__global__ void sigmoid_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values
);

__global__ void tanh_activation(
	data_t* activations, size_t activations_start, size_t layer_activation_start, short write_activation,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	size_t neuron_execution_values_read, size_t neuron_execution_values_write, short write_execution_values
);