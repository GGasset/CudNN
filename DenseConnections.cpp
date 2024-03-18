#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DenseConnections.h"
#include "connections.cu"

void DenseConnections::linear_function(data_t* activations,
	data_t* execution_values, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron,
	parameter_t* weights, size_t* weights_starts, parameter_t* biases, size_t layer_length)
{
	cud_dense_linear_function<<<layer_length, previous_layer_length>>>(
		weights_starts, weights, biases,
		previous_layer_activations_start, activations,
		execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
}
