#include "LSTMLayer.h"

void LSTMLayer::layer_specific_initialize_fields(size_t connection_count, size_t neuron_count)
{
	cudaMalloc(&state, sizeof(data_t) * neuron_count * 2);
	cudaMalloc(&neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaDeviceSynchronize();
	cudaMemset(state, 0, sizeof(data_t) * neuron_count * 2);
	IConnections::generate_random_values(&neuron_weights, neuron_count * 4);
	cudaDeviceSynchronize();
}

void LSTMLayer::execute(data_t* activations, size_t activations_start, data_t* execution_values, size_t execution_values_start)
{
	// neuron execution values 0
	connections->linear_function(
		activations_start, activations,
		execution_values, execution_values_start,
		execution_values_layer_start, execution_values_per_neuron
	);
	LSTM_execution kernel(1, neuron_count) (
		activations, activations_start, layer_activations_start,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights, state
	);
}

void LSTMLayer::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* derivatives, size_t derivatives_start,
	data_t* gradients, size_t next_gradients_start, size_t gradients_start,
	data_t* costs, size_t costs_start
)
{
	LSTM_gradient_calculation kernel(1, neuron_count) (
		derivatives, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
		gradients, gradients_start, next_gradients_start, layer_gradients_start, neuron_gradients_starts, connection_associated_gradient_counts,
		costs, costs_start, layer_activations_start
	);
	cudaDeviceSynchronize();
	connections->calculate_gradients(
		activations, activations_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start
	);
	cudaDeviceSynchronize();
}

void LSTMLayer::subtract_gradients(data_t* gradients, size_t gradients_start, data_t learning_rate, short* dropout, data_t gradient_clip)
{
	connections->subtract_gradients(
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		learning_rate, dropout, gradient_clip
	);
	LSTM_gradient_subtraction kernel(1, neuron_count) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts, connection_associated_gradient_counts,
		neuron_weights, learning_rate, dropout, gradient_clip
	);
}

void LSTMLayer::calculate_derivatives(
	data_t* activations, size_t activations_start,
	data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
	data_t* execution_values, size_t execution_values_start
)
{
	connections->calculate_derivative(
		activations_start, activations, derivatives_start, layer_derivatives_start, derivatives_per_neuron, derivatives
	);
	cudaDeviceSynchronize();
	LSTM_derivative_calculation kernel(1, neuron_count) (
		derivatives, previous_derivatives_start, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights
	);
}

void LSTMLayer::delete_memory()
{
	cudaMemset(state, 0, sizeof(data_t) * 2 * neuron_count);
	cudaDeviceSynchronize();
}

void LSTMLayer::layer_specific_deallocate()
{
	cudaFree(neuron_weights);
	cudaFree(state);
}