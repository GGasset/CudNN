#include <stdlib.h>
#include "curand.h"

#include "IConnections.h"
#include "activations.cu"

#pragma once
class ILayer
{
public:
	size_t connection_count = 0;
	size_t neuron_count = 0;
	IConnections* connections = 0;
	parameter_t* weights = 0;
	parameter_t* biases = 0;
	size_t layer_activations_start = 0;
	size_t execution_values_layer_start = 0;
	size_t execution_values_per_neuron = 0;

	void Initialize_fields(size_t connection_count, size_t neuron_count);

	void generate_random_weights(size_t connection_count, size_t start_i = 0);

	virtual void add_neuron(size_t neurons_to_add, size_t connection_count_per_neuron) = 0;

	virtual void deallocate();

	virtual void execute(
		data_t *activations, size_t activations_start,
		data_t *execution_values, size_t execution_values_start
	) = 0;
};

