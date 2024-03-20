#include <stdlib.h>
#include "curand.h"

#include "IConnections.h"
#include "activations.cu"

#pragma once
class ILayer
{
public:
	size_t neuron_count = 0;
	IConnections* connections = 0;
	parameter_t* weights = 0;
	parameter_t* biases = 0;
	size_t layer_activations_start = 0;
	size_t execution_values_layer_start = 0;
	size_t execution_values_per_neuron = 0;

	void initialize_weights(size_t connection_count)
	{
		curandGenerator_t generator;
		curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
		cudaMalloc(&weights, sizeof(parameter_t) * connection_count);
		curandGenerateUniform(generator, weights, connection_count);
	}

	virtual void deallocate()
	{
		cudaFree(weights);
		cudaFree(biases);
		connections->deallocate();
		free(connections);
	}

	virtual void execute(
		data_t *activations, size_t activations_start,
		data_t *execution_values, size_t execution_values_start
	) = 0;
};

