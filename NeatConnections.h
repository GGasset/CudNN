#pragma once
#include "IConnections.h"
class NeatConnections :
    public IConnections
{
	size_t* connection_points = 0;
	size_t* connection_counts = 0;

	void linear_function(size_t activations_start, data_t* activations,
		data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron,
		field_t* weights, field_t* biases, size_t layer_length) override;

	void calculate_derivative(
		size_t activations_start, data_t* activations,
		size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
		field_t* weights, size_t layer_length
	) override;

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		data_t* costs, size_t costs_start,
		field_t* weights, size_t layer_length
	) override;

	void subtract_gradients(
		data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
		field_t* weights, field_t* biases, size_t neuron_count,
		data_t learning_rate, short* dropout, data_t gradient_clip
	) override;

	void add_neuron(size_t neurons_to_add, size_t connections_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability) override;

};

