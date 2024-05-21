#include "costs.cuh"

__global__ void MSE_derivative(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t output_length
)
{
	costs[costs_start + neuron_count * blockIdx.x + last_layer_activations_start + threadIdx.x] = 
		-2 * (Y_hat[blockIdx.x * output_length + threadIdx.x] - activations[activations_start + last_layer_activations_start + threadIdx.x]);
}