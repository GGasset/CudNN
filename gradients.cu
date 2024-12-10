#include "gradients.cuh"

__global__ void LSTM_gradient_calculation(
	data_t* derivatives, size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron,
	data_t* gradients, size_t gradients_start, size_t next_t_gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_derivatives_start = derivatives_start + derivatives_layer_start + derivatives_per_neuron * tid;
	
	size_t connections_gradients_start = gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	size_t neuron_gradients_start = connections_gradients_start + connection_associated_gradient_counts[tid];
	
	size_t next_connections_gradients_start = next_t_gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	size_t next_neuron_gradients_start = next_connections_gradients_start + connection_associated_gradient_counts[tid];

	data_t current_gradient = costs[costs_start + layer_costs_start + tid];

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
																	// Output derivative

	gradients[neuron_gradients_start + 1] = current_gradient * derivatives[neuron_derivatives_start + 13]; // Output weight gradient
																	// output weight derivative with respect to the weight

	current_gradient *= derivatives[neuron_derivatives_start + 14];
							// output weight derivative, with respect to nothing

	linear_hidden_gradient += current_gradient *= derivatives[neuron_derivatives_start + 1];
													// sigmoid(linear + hidden state) derivative

	current_gradient = output_multiplication_gradient;
	current_gradient *= derivatives[neuron_derivatives_start + 12];
							// tanh(output cell state) derivative

	data_t initial_cell_state_gradient = current_gradient += next_cell_state_gradient;


	// Store gate
	data_t cell_state_addition_gradient = current_gradient *= derivatives[neuron_derivatives_start + 11];
																// cell state addition derivative

	data_t store_gate_multiplication_gradient = current_gradient *= derivatives[neuron_derivatives_start + 10];
																		// store gate multiplication derivative

	gradients[neuron_gradients_start + 2] = current_gradient * derivatives[neuron_derivatives_start + 8]; // Tanh weight gradient
																// store gate tanh weight derivative with respect to the weight

	gradients[neuron_gradients_start + 3] = current_gradient * derivatives[neuron_derivatives_start + 6]; // Sigmoid weight gradient
																// store gate sigmoid weight derivative with respect to the weight

	linear_hidden_gradient += current_gradient * derivatives[neuron_derivatives_start + 9] * derivatives[neuron_derivatives_start + 2];
	linear_hidden_gradient += current_gradient * derivatives[neuron_derivatives_start + 7] * derivatives[neuron_derivatives_start + 1];

	// Forget gate
	current_gradient = cell_state_addition_gradient;
	current_gradient *= derivatives[neuron_derivatives_start + 5];
	gradients[neuron_gradients_start + 6] = current_gradient * derivatives[neuron_derivatives_start + 3]; // Forget weight gradient
	gradients[neuron_gradients_start + 7] = //cell_state_addition_gradient * derivatives[neuron_derivatives_start + 16]; // Inital cell state gradient (cell state multiplication gradient)
		initial_cell_state_gradient * derivatives[neuron_derivatives_start + 10] * derivatives[neuron_derivatives_start + 16];
	current_gradient = linear_hidden_gradient += current_gradient * derivatives[neuron_derivatives_start + 4] * derivatives[neuron_derivatives_start + 1];
	current_gradient *= derivatives[neuron_derivatives_start];
	gradients[connections_gradients_start] = current_gradient; // Linear Function gradient | Initial Hidden State gradient
}

__global__ void LSTM_gradient_subtraction(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, size_t* connection_associated_gradient_counts,
	field_t* neuron_weights,
	data_t learning_rate, short* dropout, data_t max_subtracted_gradient,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t neuron_i = tid;
	size_t neuron_gradients_start_i = gradients_start + layer_gradients_start + neuron_gradients_starts[tid] + connection_associated_gradient_counts[tid];
	size_t neuron_weights_start = static_cast<size_t>(4) * tid;

	neuron_weights[neuron_weights_start] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 6] * learning_rate * dropout[neuron_i]); // Forget weight
	neuron_weights[neuron_weights_start + 1] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 3] * learning_rate * dropout[neuron_i]); // Store sigmoid weight
	neuron_weights[neuron_weights_start + 2] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 2] * learning_rate * dropout[neuron_i]); // Store Tanh weight
	neuron_weights[neuron_weights_start + 3] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 1] * learning_rate * dropout[neuron_i]); // Output_weight
}

__global__ void neuron_gradient_calculation(
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start,
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts,
	data_t* costs, size_t costs_start, size_t layer_costs_start,
	ActivationFunctions activation,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	data_t input_gradient = costs[costs_start + layer_costs_start + tid];
	data_t activation_input = execution_values[execution_values_start + execution_values_layer_start + tid];
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
	size_t gradient_write_i = gradients_start + layer_gradients_start + neuron_gradients_starts[tid];
	gradients[gradient_write_i] = bias_gradient;
}

__global__ void cud_set_dropout(
	float dropout_rate, float* normalized_random_samples, short* dropout,
	size_t layer_length
)
{
	size_t tid = get_tid();
	if (tid >= layer_length) return;

	size_t i = tid;
	dropout[i] = normalized_random_samples[i] > dropout_rate;
}
