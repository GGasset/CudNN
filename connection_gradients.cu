#include "connection_gradients.cuh"

__global__ void cud_dense_gradient_calculation(
	data_t* activations, size_t activations_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start,
	size_t previous_layer_activations_start,
	field_t* weights
)
{
	// Input gradient is bias gradient
	size_t input_gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[blockIdx.x];
	data_t input_gradient = gradients[input_gradient_i];
	size_t weight_gradient_i = input_gradient_i + threadIdx.x + 1;
	field_t weight = weights[threadIdx.x];
	data_t activation = activations[activations_start + previous_layer_activations_start + threadIdx.x];
	gradients[weight_gradient_i] = input_gradient * activation;
	costs[costs_start + previous_layer_activations_start + threadIdx.x] -= input_gradient * weight;
}