#include "derivatives.cuh"

#include <cmath>

__device__ data_t device_sigmoid_derivative(
	data_t input
)
{
	data_t exp_minus_x = exp(-input);
	return (exp_minus_x) / ((1 +exp_minus_x) * (1 + exp_minus_x));
}

__device__ data_t device_tanh_derivative(
	data_t input
)
{
	data_t exp_2_x = exp(input * 2);
	return (4 * exp_2_x) / ((exp_2_x + 1) * (exp_2_x + 1));
}

__global__ void LSTM_derivative_calculation(
	data_t* prev_state_derivatives, data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights, 
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_execution_values_start = execution_values_start + execution_values_layer_start + execution_values_per_neuron * tid;

	size_t neuron_weights_start = static_cast<size_t>(4) * tid;

	size_t neuron_derivatives_start = derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;
	size_t previous_neuron_derivatives_start = previous_derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;

	data_t linear_function_derivative = derivatives[neuron_derivatives_start];

	data_t previous_hidden_derivative = prev_state_derivatives[tid * 2];
	data_t previous_cell_derivative = prev_state_derivatives[tid * 2 + 1];
	if (derivatives_start != 0)
	{
		previous_hidden_derivative = derivatives[previous_neuron_derivatives_start + 15];
		previous_cell_derivative = derivatives[previous_neuron_derivatives_start + 11];
	}
	derivatives[neuron_derivatives_start + 17] = previous_hidden_derivative;
	
	derivatives[neuron_derivatives_start] = linear_function_derivative + previous_hidden_derivative;

	data_t linear_hidden = execution_values[neuron_execution_values_start];
	
	data_t linear_hidden_sigmoid_derivative = derivatives[neuron_derivatives_start + 1] = device_sigmoid_derivative(linear_hidden);
	data_t linear_hidden_tanh_derivative = derivatives[neuron_derivatives_start + 2] = device_tanh_derivative(linear_hidden);

	data_t linear_hidden_sigmoid = execution_values[neuron_execution_values_start + 1];
	data_t linear_hidden_tanh = execution_values[neuron_execution_values_start + 5];

	// Forget gate
	data_t forget_weight = neuron_weights[neuron_weights_start];
	data_t forget_weight_derivative = derivatives[neuron_derivatives_start + 3] = 
		linear_hidden_sigmoid;

	data_t forget_weight_multiplication_derivative = derivatives[neuron_derivatives_start + 4] =
		linear_hidden_sigmoid + forget_weight * linear_hidden_sigmoid_derivative;

	data_t initial_cell_state = execution_values[neuron_execution_values_start + 3];
	data_t forget_gate_weight_multiplication = execution_values[neuron_execution_values_start + 2];
	data_t cell_state_multiplication_derivative = derivatives[neuron_derivatives_start + 5] = 
		previous_cell_derivative * forget_gate_weight_multiplication + forget_weight_multiplication_derivative * initial_cell_state;

	derivatives[neuron_derivatives_start + 16] = // Cell state multiplication derivative with respect of the previous state
		forget_gate_weight_multiplication; //+ //initial_cell_state * forget_weight_multiplication_derivative;

	// Store gate
	data_t store_sigmoid_weight = neuron_weights[neuron_weights_start + 1];
	data_t store_tanh_weight = neuron_weights[neuron_weights_start + 2];
	
	data_t store_gate_sigmoid_weight_derivative = derivatives[neuron_derivatives_start + 6] =
		linear_hidden_sigmoid;

	data_t store_gate_sigmoid_weight_multiplication_derivative = derivatives[neuron_derivatives_start + 7] =
		linear_hidden_sigmoid + store_sigmoid_weight * linear_hidden_sigmoid_derivative;

	data_t store_gate_tanh_weight_derivative = derivatives[neuron_derivatives_start + 8] =
		linear_hidden_tanh;

	data_t store_gate_tanh_weight_multiplication_derivative = derivatives[neuron_derivatives_start + 9] =
		linear_hidden_tanh + store_tanh_weight * linear_hidden_tanh_derivative;

	data_t store_gate_sigmoid_weight_multiplication = execution_values[neuron_execution_values_start + 4];
	data_t store_gate_tanh_weight_multiplication = execution_values[neuron_execution_values_start + 6];

	data_t store_gate_multiplication_derivative = derivatives[neuron_derivatives_start + 10] =
		store_gate_sigmoid_weight_multiplication_derivative * store_gate_tanh_weight_multiplication + store_gate_sigmoid_weight_multiplication * store_gate_tanh_weight_multiplication_derivative;

	data_t cell_state_addition_derivative = derivatives[neuron_derivatives_start + 11] =
		store_gate_multiplication_derivative + cell_state_multiplication_derivative;

	//derivatives[neuron_derivatives_start + 16] = store_gate_multiplication_derivative

	// Output gate
	data_t cell_state_tanh = execution_values[neuron_execution_values_start + 8];
	data_t output_weight = neuron_weights[neuron_weights_start + 3];
	data_t output_weight_multiplication = execution_values[neuron_execution_values_start + 9];

	data_t cell_state_tanh_derivative = derivatives[neuron_derivatives_start + 12] = 
		device_tanh_derivative(execution_values[neuron_execution_values_start + 7]);

	data_t output_gate_weight_derivative = derivatives[neuron_derivatives_start + 13] =
		linear_hidden_sigmoid;

	data_t output_gate_weight_multiplication_derivative = derivatives[neuron_derivatives_start + 14] =
		linear_hidden_sigmoid + output_weight * linear_hidden_sigmoid_derivative;

	data_t output_derivative = derivatives[neuron_derivatives_start + 15] =
		cell_state_tanh_derivative * output_weight_multiplication + cell_state_tanh * output_gate_weight_multiplication_derivative;

	prev_state_derivatives[tid * 2] = output_derivative;
	prev_state_derivatives[tid * 2 + 1] = cell_state_addition_derivative;
}

