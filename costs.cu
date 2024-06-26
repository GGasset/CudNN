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

	data_t predicted = activations[activations_start + neuron_count * t + last_layer_activations_start + tid];
	data_t Y = Y_hat[output_length * t + tid]; 
	//data_t derivative = -2 * (Y_hat[output_length * t + tid] - activations[activations_start + neuron_count * t + last_layer_activations_start + tid]);
	data_t derivative = 2 * (predicted - Y);
	costs[costs_start + t * neuron_count + last_layer_activations_start + tid] = derivative;
}

__global__ void MSE_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* Y_hat, size_t output_length,
	data_t* cost_write
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t predicted = activations[activations_start + neuron_count * t + last_layer_activations_start + tid];
	data_t Y = Y_hat[output_length * t + tid];
	data_t error = Y - predicted;
	error *= error;
	atomicAdd(cost_write, error);
}

__global__ void output_over_reward_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* rewards, size_t output_length,
	data_t* cost
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;
	
	data_t reward = rewards[t];
	data_t prediction = activations[activations_start + neuron_count * t + last_layer_activations_start + tid];
	data_t output = 0;
	if (reward) output = (prediction + 3) / reward;
	
	atomicAdd(cost, output);
}

__global__ void output_over_reward_derivative(
	data_t* costs, size_t costs_start,
	size_t neuron_count, size_t last_layer_activations_start, size_t output_length,
	data_t* rewards
)
{
	size_t tid = get_tid();
	if (tid >= output_length) return;
	size_t t = blockIdx.y;

	data_t reward = rewards[t];
	data_t cost_derivative = 0;
	if (reward) cost_derivative = 1 / reward;
	
	size_t cost_write = costs_start + neuron_count * t + last_layer_activations_start;
	costs[cost_write] = cost_derivative;
}
