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


	data_t next_hidden_state_gradient = 0;
	data_t next_cell_state_gradient = 0;
	if (next_t_gradients_start)
	{
		next_hidden_state_gradient = gradients[next_neuron_gradients_start + 5];
		next_cell_state_gradient = gradients[next_neuron_gradients_start + 4];
	}

	data_t sigmoid_lh_derivative = derivatives[neuron_derivatives_start + 5];
	data_t tanh_lh_derivative = derivatives[neuron_derivatives_start + 6];



	// Output Losses
	data_t output_gradient = costs[costs_start + layer_costs_start + tid];
	data_t output_hidden_gradient = output_gradient + next_hidden_state_gradient;



	// Output Gate
	data_t output_multiplication_to_weight_gradient = output_hidden_gradient * derivatives[neuron_derivatives_start + 21];
																		
	data_t output_weight_gradient = output_multiplication_to_weight_gradient * derivatives[neuron_derivatives_start + 19];
	gradients[neuron_gradients_start + 3] = output_weight_gradient;

	data_t output_sigmoid_gradient = output_multiplication_to_weight_gradient * derivatives[neuron_derivatives_start + 20] * sigmoid_lh_derivative;

	data_t output_multiplication_to_tanh_gradient = output_hidden_gradient * derivatives[neuron_derivatives_start + 22];
	data_t cell_state_tanh_gradient = output_multiplication_to_tanh_gradient * derivatives[neuron_derivatives_start + 23];
	data_t output_cell_state_gradient = cell_state_tanh_gradient - next_cell_state_gradient;



	// Store gate
	//  Cell state
	data_t store_addition_to_cell_state_grad = output_cell_state_gradient * derivatives[neuron_derivatives_start + 18];
	data_t store_addition_to_store_gate_grad = output_cell_state_gradient * derivatives[neuron_derivatives_start + 17];

	//  Candidate cell state gate
	data_t candidate_output_multiplication_gradient = store_addition_to_store_gate_grad * derivatives[neuron_derivatives_start + 16];
	data_t candidate_weight_gradient = candidate_output_multiplication_gradient * derivatives[neuron_derivatives_start + 13];
	gradients[neuron_gradients_start + 2] = candidate_weight_gradient;

	data_t candidate_tanh_gradient = candidate_output_multiplication_gradient * derivatives[neuron_derivatives_start + 14] * tanh_lh_derivative;

	//  Input gate
	data_t input_gate_output_multiplication_gradient = store_addition_to_store_gate_grad * derivatives[neuron_derivatives_start + 15];
	data_t input_weight_gradient = input_gate_output_multiplication_gradient * derivatives[neuron_derivatives_start + 11];
	gradients[neuron_gradients_start + 1] = input_weight_gradient;

	data_t input_sigmoid_gradient = input_gate_output_multiplication_gradient * derivatives[neuron_derivatives_start + 12] * sigmoid_lh_derivative;



	// Forget gate
	data_t forget_multiplication_to_forget_gate = store_addition_to_cell_state_grad * derivatives[neuron_derivatives_start + 9];
	data_t initial_cell_gradient = store_addition_to_cell_state_grad * derivatives[neuron_derivatives_start + 10];
	gradients[neuron_gradients_start + 4] = initial_cell_gradient;

	data_t forget_weight_gradient = forget_multiplication_to_forget_gate * derivatives[neuron_derivatives_start + 7];
	gradients[neuron_gradients_start] = forget_weight_gradient;

	data_t forget_sigmoid_gradient = forget_multiplication_to_forget_gate * derivatives[neuron_derivatives_start + 8] * sigmoid_lh_derivative;



	// Input gradients
	data_t linear_hidden_gradient = output_sigmoid_gradient + candidate_tanh_gradient + input_sigmoid_gradient + forget_sigmoid_gradient;
	data_t hidden_state_gradient = linear_hidden_gradient * derivatives[neuron_derivatives_start + 1];
	data_t linear_function_gradient = linear_hidden_gradient * derivatives[neuron_derivatives_start];

	gradients[neuron_gradients_start + 5] = -hidden_state_gradient;
	gradients[connections_gradients_start] = linear_function_gradient;
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

	neuron_weights[neuron_weights_start] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i] * learning_rate * dropout[neuron_i]); // Forget weight
	neuron_weights[neuron_weights_start + 1] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 1] * learning_rate * dropout[neuron_i]); // Store sigmoid weight
	neuron_weights[neuron_weights_start + 2] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 2] * learning_rate * dropout[neuron_i]); // Store Tanh weight
	neuron_weights[neuron_weights_start + 3] -= device_closest_to_zero(max_subtracted_gradient, gradients[neuron_gradients_start_i + 3] * learning_rate * dropout[neuron_i]); // Output_weight
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
