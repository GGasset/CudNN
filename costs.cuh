#include <bits/stdc++.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"
#include "cuda_functionality.cuh"

enum CostFunctions
{
	MSE,
	output_over_reward
};

__global__ void MSE_derivative(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t output_length
);

__global__ void MSE_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* Y_hat, size_t output_length,
	data_t* cost_write
);

__global__ void output_over_reward_derivative(
	data_t* costs, size_t costs_start,
	size_t neuron_count, size_t last_layer_activations_start, size_t output_length,
	data_t* rewards
);

__global__ void output_over_reward_cost(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* rewards, size_t output_length,
	data_t* cost
);
