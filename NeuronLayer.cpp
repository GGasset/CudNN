#ifndef NEURONLAYER_DEFINITIONS
#define NEURONLAYER_DEFINITIONS

#include "NeuronLayer.h"

void NeuronLayer::execute(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start
)
{
	connections->linear_function(activations_start, activations,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		weights, biases,
		neuron_count);
	cudaDeviceSynchronize();
	switch (activation)
	{
	case sigmoid:
		sigmoid_activation kernel(1, neuron_count) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0
		);
		break;
	default:
		break;
	}
	cudaDeviceSynchronize();
}

#endif