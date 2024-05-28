#include "connection_gradients.cuh"

__global__ void cud_dense_gradient_calculation(
	data_t* activations, size_t activations_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start,
	size_t previous_layer_activations_start, size_t previous_layer_length,
	field_t* weights
)
{
	size_t tid = get_tid();
	if (tid >= previous_layer_length)
		return;

	// Input gradient is bias gradient
	size_t input_gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[blockIdx.y];
	data_t input_gradient = gradients[input_gradient_i];
	size_t weight_gradient_i = input_gradient_i + tid + 1;
	field_t weight = weights[tid];
	data_t activation = activations[activations_start + previous_layer_activations_start + tid];
	gradients[weight_gradient_i] = input_gradient * activation;
	atomicAdd(costs + costs_start + previous_layer_activations_start + tid, -input_gradient * weight);
}

__global__ void bias_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	field_t* biases, data_t learning_rate, short* dropout, data_t max_subtracted_gradient
)
{
	size_t gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[threadIdx.x];
	biases[threadIdx.x] -= device_min(max_subtracted_gradient, gradients[gradient_i] * learning_rate * dropout[threadIdx.x]);
}

__global__ void cud_dense_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	field_t* weights, size_t previous_layer_length,
	data_t learning_rate, short* dropout, data_t max_subtracted_gradient
)
{
	size_t gradient_i = gradients_start + layer_gradients_start + neuron_gradients_starts[blockIdx.x] + threadIdx.x + 1;
	size_t weight_i = previous_layer_length * blockIdx.x + threadIdx.x;
	atomicAdd(weights + weight_i, -device_min(max_subtracted_gradient, gradients[gradient_i] * learning_rate * dropout[blockIdx.x]));
}