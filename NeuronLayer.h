#include "ILayer.h"

#pragma once
class NeuronLayer : public ILayer
{
protected:
	ActivationFunctions activation = ActivationFunctions::sigmoid;

public:
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

	void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start
	) override
	{

	}
};

