#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

#include "data_type.h"
#include "kernel_macros.h"
#include "linear_functions.cuh"
#include "connection_gradients.cuh"

#pragma once
class IConnections
{
public:
	field_t* weights = 0;
	field_t* biases = 0;
	size_t neuron_count = 0;

	static void generate_random_values(float** pointer, size_t float_count, size_t start_i = 0);

	virtual void linear_function(
		size_t activations_start, data_t* activations,
		data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
		) = 0;

	virtual void calculate_derivative(
		size_t activations_start, data_t* activations,
		size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
	) = 0;

	virtual void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t* costs, size_t costs_start
	) = 0;

	virtual void subtract_gradients(
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t learning_rate, short* dropout, data_t gradient_clip
	) = 0;

	virtual void add_neuron(size_t neurons_to_add, size_t connections_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability = 1)
	{

	}

	void deallocate();
	virtual void specific_deallocate();
};

