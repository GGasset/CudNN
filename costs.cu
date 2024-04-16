#include "costs.cuh"

__global__ void MSE_derivative(
	data_t* activations, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t Y_hat_start
)
{
	costs[costs_start + last_layer_activations_start + threadIdx.x] = 
		2 * (Y_hat[Y_hat_start + threadIdx.x] - activations[activations_start + last_layer_activations_start + threadIdx.x]);
}