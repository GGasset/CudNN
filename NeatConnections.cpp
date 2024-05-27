#include "NeatConnections.h"

void NeatConnections::linear_function(size_t activations_start, data_t* activations, data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron, field_t* weights, field_t* biases, size_t layer_length)
{
	size_t connections_start = 0;
	for (size_t i = 0; i < layer_length; i++)
	{
		size_t connection_count = connection_counts[i];
		dim3 gridDim = dim3(connection_counts[i] / 32 + (connection_counts[i] % 32 > 0), 1, 1);
		cud_NEAT_neuron_linear_function kernel(gridDim, 32) (
			i, connection_count, weights, connection_points, connections_start,
		activations_start, activations,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
		connections_start += connection_count;
	}
	cud_add_biases kernel(dim3(layer_length / 32 + (layer_length % 32 > 0), 1, 1), 32) (
		layer_length, biases, 
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cudaDeviceSynchronize();
}
