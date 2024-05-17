#include "ILayer.h"

#pragma once
class LSTMLayer : public ILayer
{
public:
	field_t* neuron_weights = 0;
	data_t* state = 0;

	void layer_specific_initialize_fields(size_t connection_count, size_t neuron_count) override;
	void layer_specific_deallocate() override;

	void execute(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start
	) override;

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* derivatives, size_t derivatives_start,
		data_t* gradients, size_t next_gradients_start, size_t gradients_start,
		data_t* costs, size_t costs_start
	) override;

	void subtract_gradients(
		data_t* gradients, size_t gradients_start
	) override;

	void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
		data_t* execution_values, size_t execution_values_start
	) override;

	void delete_memory() override;
};

