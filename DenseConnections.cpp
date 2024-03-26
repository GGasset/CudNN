#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DenseConnections.h"
#include "connections.cu"

DenseConnections::DenseConnections(size_t previous_layer_activations_start, size_t previous_layer_length)
{
	this->previous_layer_activations_start = previous_layer_activations_start;
	this->previous_layer_length = previous_layer_length;
}

void DenseConnections::linear_function(size_t activations_start, data_t* activations,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron,
	parameter_t* weights, parameter_t* biases, size_t layer_length)
{
	cud_dense_linear_function kernel(layer_length, previous_layer_length) (
		previous_layer_length, weights,
		activations_start, previous_layer_activations_start, activations,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cud_add_biases kernel(1, layer_length) (
		biases,
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
}
