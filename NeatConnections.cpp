#include "NeatConnections.h"

void NeatConnections::linear_function(size_t activations_start, data_t* activations, data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, field_t* weights, field_t* biases, size_t layer_length)
{
	dim3 gridDim = dim3(connection_count / 32 + (connection_count % 32 > 0), layer_length, 1);
	cud_NEAT_linear_function kernel(gridDim, 32) (
		connection_count, weights, connection_points,
		activations_start, activations,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cud_add_biases kernel(dim3(layer_length / 32 + (layer_length % 32 > 0), 1, 1), 32) (
		layer_length, biases, 
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
}
