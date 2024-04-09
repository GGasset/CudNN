#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "IConnections.h"

#pragma once
class DenseConnections : public IConnections
{
public:
	size_t previous_layer_activations_start = 0;
	size_t previous_layer_length = 0;

	DenseConnections(size_t previous_layer_activations_start, size_t previous_layer_length);

	void linear_function(size_t activations_start, data_t* activations,
		data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron,
		field_t* weights, field_t* biases, size_t layer_length) override;

	void calculate_derivative(
		size_t activations_start, data_t* activations,
		size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives,
		field_t* weights, size_t layer_length
	) override;

	void add_neuron(size_t neurons_to_add, size_t connections_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability) override;
};

