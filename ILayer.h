#include <stdlib.h>
#include "curand.h"

#include "IConnections.h"
#include "neuron_operations.cuh"
#include "derivatives.cuh"
#include "gradients.cuh"

#pragma once
class ILayer
{
protected:
	/// <summary>
	/// Modify through set neuron count
	/// </summary>
	size_t neuron_count = 0;

public:
	IConnections* connections = 0;

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

	size_t get_neuron_count();
	void set_neuron_count(size_t neuron_count);

	void initialize_fields(size_t connection_count, size_t neuron_count);
	virtual void layer_specific_initialize_fields(size_t connection_count, size_t neuron_count);

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
		data_t* gradients, size_t gradients_start, data_t learning_rate, short* dropout, data_t gradient_clip
	) = 0;

	virtual void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
		data_t* execution_values, size_t execution_values_start
	) = 0;

	virtual void add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections) = 0;
	virtual void adjust_to_added_neuron(size_t added_neuron_i, float connection_probability) = 0;
	virtual void remove_neuron(size_t layer_neuron_i) = 0;
	virtual void adjust_to_removed_neuron(size_t neuron_i) = 0;

	virtual void delete_memory();
};

