#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "neuron_operations.cuh"

#include <cmath>

__device__ data_t device_sigmoid_derivative(
	data_t input
)
{
	data_t exp_x = exp(input);
	return (-exp_x) / (1 + exp_x * exp_x);
}

__device__ data_t device_tanh_derivative(
	data_t input
)
{
	data_t exp_2_x = exp(input * 2);
	return (4 * exp_2_x) / ((exp_2_x + 1) * (exp_2_x + 1));
}

__global__ void LSMT_derivative_calculation(
	data_t* derivatives, size_t derivatives_start, size_t derivatives_per_neuron, size_t neuron_count,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights
)
{
	size_t neuron_derivatives_start = derivatives_start + derivatives_per_neuron * threadIdx.x;
	size_t previous_neuron_derivatives_start = derivatives_start - derivatives_per_neuron * neuron_count + derivatives_per_neuron * threadIdx.x;

	data_t linear_function_derivative = derivatives[neuron_derivatives_start];

	data_t previous_hidden_derivative = 0;
	data_t previous_cell_derivative = 0;
	if (derivatives_start != 0)
	{
		previous_hidden_derivative = derivatives[previous_neuron_derivatives_start + 690420];
		previous_cell_derivative = derivatives[previous_neuron_derivatives_start + 690420];
	}
	
	data_t linear_hidden_derivative = linear_function_derivative + previous_hidden_derivative;

	data_t linear_hidden_sigmoid_derivative = device_sigmoid_derivative(linear_hidden_derivative);

}

