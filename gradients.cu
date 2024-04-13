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
	current_gradient *= derivatives[neuron_gradients_start + 12];
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