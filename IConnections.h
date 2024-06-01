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
	/// <summary>
	/// Device Array
	/// </summary>
	field_t* weights = 0;
	/// <summary>
	/// Device Array
	/// </summary>
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

	virtual void add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections);
	virtual void adjust_to_added_neuron(size_t added_neuron_i, float connection_probability);
	virtual void remove_neuron(size_t layer_neuron_i);
	virtual void adjust_to_removed_neuron(size_t neuron_i);

	void deallocate();
	virtual void specific_deallocate();
};

