#include "LSMTLayer.h"

void LSMTLayer::execute(data_t* activations, size_t activations_start, data_t* execution_values, size_t execution_values_start)
{
	// neuron execution values 0
	connections->linear_function(
		activations_start, activations,
		execution_values, execution_values_start,
		execution_values_layer_start, execution_values_per_neuron,
		weights, biases, neuron_count
	);
	LSTM_execution kernel(1, neuron_count) (
		activations, activations_start, layer_activations_start,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights, state;
	);
}

void LSMTLayer::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start, short calculate_derivatives,
	data_t* gradients, size_t next_gradients_start, size_t gradients_start,
	data_t* costs, size_t costs_start
)
{
	if (calculate_derivatives)
	{
		connections->calculate_derivative(
			activations_start, activations,  derivatives_start, layer_derivatives_start, derivatives_per_neuron, derivatives,
			weights, neuron_count
		);
		LSTM_derivative_calculation kernel(1, neuron_count) (
			derivatives, previous_derivatives_start, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
			neuron_weights
		);
	}
	LSTM_gradient_calculation kernel(1, neuron_count) (
		derivatives, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
		gradients, gradients_start, next_gradients_start, layer_gradients_start, neuron_gradients_starts, connection_associated_gradient_counts,
		costs, costs_start, layer_activations_start
	);
	connections->calculate_gradients(
		activations, activations_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start
	);
}