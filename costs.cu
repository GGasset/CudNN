#include "costs.cuh"

__global__ void MSE_derivative(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t output_length
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t derivative = -2 * (Y_hat[output_length * t + tid] - activations[activations_start + neuron_count * t + last_layer_activations_start + tid]);
	costs[costs_start + t * neuron_count + last_layer_activations_start + tid] = derivative;
}
