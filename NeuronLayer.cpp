#include "NeuronLayer.h"

void ILayer::execute(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start
)
{
	connections->linear_function(activations_start, activations,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		weights, biases,
		neuron_count);
}