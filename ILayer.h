#include <stdlib.h>
#include "curand.h"

#include "IConnections.h"
#include "neuron_operations.cuh"
#include "derivatives.cuh"
#include "gradients.cuh"

#pragma once
class ILayer
{
public:
	size_t connection_count = 0;
	size_t neuron_count = 0;
	IConnections* connections = 0;
	field_t* weights = 0;
	field_t* biases = 0;

	size_t layer_activations_start = 0;

	size_t execution_values_layer_start = 0;
	size_t execution_values_per_neuron = 0;

	size_t layer_derivative_count = 0;
	size_t layer_derivatives_start = 0;
	size_t derivatives_per_neuron = 0;
	
	size_t layer_gradient_count = 0;
	size_t layer_gradients_start = 0;
	size_t* neuron_gradients_starts = 0;
	size_t* connection_associated_gradient_counts = 0;

	void initialize_fields(size_t connection_count, size_t neuron_count);
	virtual void layer_specific_initialize_fields(size_t connection_count, size_t neuron_count);

	static void generate_random_values(float** pointer, size_t float_count, size_t start_i = 0);

	virtual void add_neuron(size_t neurons_to_add, size_t connection_count_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability = 1);

	void deallocate();

	virtual void layer_specific_deallocate();

	virtual void execute(
		data_t *activations, size_t activations_start,
		data_t *execution_values, size_t execution_values_start
	) = 0;

	virtual void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* derivatives, size_t derivatives_start,
		data_t* gradients, size_t next_gradients_start, size_t gradients_start,
		data_t* costs, size_t costs_start
	) = 0;

	virtual void subtract_gradients(
		data_t* gradients, size_t gradients_start, data_t learning_rate
	) = 0;

	virtual void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
		data_t* execution_values, size_t execution_values_start
	) = 0;

	virtual void delete_memory();
};

