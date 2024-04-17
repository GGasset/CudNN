#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

enum CostFunctions
{
	MSE
};

__global__ void MSE_derivative(
	data_t* activations, size_t neuron_count, size_t activations_start, size_t last_layer_activations_start,
	data_t* costs, size_t costs_start,
	data_t* Y_hat, size_t output_length
);