#pragma once

#include "gradients.cuh"

__global__ void LSTM_gradient_calculation(
	data_t* derivatives, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* gradients, size_t gradients_start, size_t next_t_gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	data_t* costs, size_t costs_start, size_t layer_costs_start
)
{
	size_t neuron_derivatives_start = derivatives_start + derivatives_layer_start + derivatives_per_neuron * threadIdx.x;
	size_t connections_gradients_start = gradients_start + layer_gradients_start + neuron_gradients_starts[threadIdx.x];
	size_t neuron_gradients_start = connections_gradients_start + connection_associated_gradient_counts[threadIdx.x];
	size_t next_neuron_gradients_start = next_t_gradients_start + layer_gradients_start + neuron_gradients_starts[threadIdx.x];

	data_t current_gradient = costs[costs_start + layer_costs_start + threadIdx.x];

	data_t next_hidden_state_gradient = 0;
	data_t next_cell_state_gradient = 0;
	if (next_t_gradients_start)
	{
		next_hidden_state_gradient = gradients[next_neuron_gradients_start];
		next_cell_state_gradient = gradients[next_neuron_gradients_start + 7];
	}
	current_gradient += next_hidden_state_gradient;

	data_t linear_hidden_gradient = 0;

	// Output gate
	data_t output_multiplication_gradient = current_gradient *= derivatives[neuron_derivatives_start + 15];
	gradients[neuron_gradients_start + 1] = current_gradient * derivatives[neuron_derivatives_start + 13]; // Output weight gradient
	current_gradient *= derivatives[neuron_derivatives_start + 14];
	linear_hidden_gradient += current_gradient *= derivatives[neuron_derivatives_start + 1];

	current_gradient = output_multiplication_gradient;
	current_gradient *= derivatives[neuron_derivatives_start + 12];
	current_gradient += next_cell_state_gradient;

	// Store gate
	data_t cell_state_addition_gradient = current_gradient *= derivatives[neuron_derivatives_start + 11];
	data_t store_gate_multiplication_gradient = current_gradient *= derivatives[neuron_derivatives_start + 10];
	gradients[neuron_gradients_start + 2] = current_gradient * derivatives[neuron_derivatives_start + 8]; // Tanh weight gradient
	gradients[neuron_gradients_start + 3] = current_gradient * derivatives[neuron_derivatives_start + 6]; // Sigmoid weight gradient

	linear_hidden_gradient += current_gradient * derivatives[neuron_derivatives_start + 9] * derivatives[neuron_derivatives_start + 2];
	linear_hidden_gradient += current_gradient * derivatives[neuron_derivatives_start + 7] * derivatives[neuron_derivatives_start + 1];

	// Forget gate
	current_gradient = cell_state_addition_gradient;
	current_gradient *= derivatives[neuron_derivatives_start + 5];
	gradients[neuron_gradients_start + 6] = current_gradient * derivatives[neuron_derivatives_start + 3]; // Forget weight gradient
	gradients[neuron_gradients_start + 7] = current_gradient; // Inital cell state gradient (cell state multiplication gradient)

	current_gradient = linear_hidden_gradient += current_gradient * derivatives[neuron_derivatives_start + 4] * derivatives[neuron_derivatives_start + 1];
	current_gradient *= derivatives[neuron_derivatives_start];
	gradients[connections_gradients_start] = current_gradient; // Linear Function gradient | Initial Hidden State gradient
}

__global__ void LSTM_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	field_t* neuron_weights,
	data_t learning_rate, short* dropout, data_t max_subtracted_gradient
)
{
	size_t neuron_i = threadIdx.x;
	size_t neuron_gradients_start = gradients_start + layer_gradients_start + neuron_gradients_starts[threadIdx.x] + connection_associated_gradient_counts[threadIdx.x];
	size_t neuron_weights_start = static_cast<size_t>(4) * threadIdx.x;

	neuron_weights[neuron_weights_start] -= device_min(max_subtracted_gradient, gradients[neuron_gradients_start + 6] * learning_rate * dropout[neuron_i]); // Forget weight
	neuron_weights[neuron_weights_start + 1] -= device_min(max_subtracted_gradient, gradients[neuron_gradients_start + 3] * learning_rate * dropout[neuron_i]); // Store sigmoid weight
	neuron_weights[neuron_weights_start + 2] -= device_min(max_subtracted_gradient, gradients[neuron_gradients_start + 2] * learning_rate * dropout[neuron_i]); // Store Tanh weight
	neuron_weights[neuron_weights_start + 3] -= device_min(max_subtracted_gradient, gradients[neuron_gradients_start + 1] * learning_rate * dropout[neuron_i]); // Output_weight
}

__global__ void neuron_gradient_calculation(
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	ActivationFunctions activation
)
{
	data_t input_gradient = -costs[costs_start + layer_costs_start + threadIdx.x];
	data_t activation_input = execution_values[execution_values_start + execution_values_layer_start + threadIdx.x];
	data_t bias_gradient = input_gradient;
	switch (activation)
	{
	case sigmoid:
		bias_gradient *= device_sigmoid_derivative(activation_input);
		break;
	case _tanh:
		bias_gradient *= device_tanh_derivative(activation_input);
		break;
	default:
		break;
	}
	size_t gradient_write_i = gradients_start + layer_gradients_start + neuron_gradients_starts[threadIdx.x];
	gradients[gradient_write_i] = bias_gradient;
}

__global__ void cud_set_dropout(
	float dropout_rate, float* normalized_random_samples, short* dropout
)
{
	size_t i = threadIdx.x;
	dropout[i] = normalized_random_samples[i] < dropout_rate;
}
