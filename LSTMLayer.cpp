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

void LSTMLayer::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
	size_t added_connection_count = connections->connection_count;
	connections->add_neuron(previous_layer_length, previous_layer_activations_start, previous_layer_connection_probability, min_connections);
	added_connection_count = connections->connection_count - added_connection_count;
	set_neuron_count(neuron_count + 1);
	
	field_t* tmp_neuron_weights = 0;
	data_t* tmp_state = 0;
	size_t* tmp_neuron_gradients_starts = new size_t[neuron_count];
	size_t* tmp_connection_associated_gradient_counts = new size_t[neuron_count];
	
	cudaMalloc(&tmp_neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaMalloc(&tmp_state, sizeof(data_t) * neuron_count * 2);
	cudaDeviceSynchronize();

	cudaMemcpy(tmp_neuron_weights, neuron_weights, sizeof(field_t) * (neuron_count - 1) * 4, cudaMemcpyDeviceToDevice);
	IConnections::generate_random_values(&tmp_neuron_weights, 4, (neuron_count - 1) * 4);
	
	cudaMemcpy(tmp_state, state, sizeof(data_t) * (neuron_count - 1) * 2, cudaMemcpyDeviceToDevice);
	cudaMemset(tmp_state + (neuron_count - 1) * 4, 0, sizeof(data_t) * 4);

	cudaMemcpy(tmp_neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count - 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count - 1, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	tmp_neuron_gradients_starts[neuron_count - 1] = tmp_neuron_gradients_starts[neuron_count - 2] + connection_associated_gradient_counts[neuron_count - 2] + 7;
	connection_associated_gradient_counts[neuron_count - 1] = added_connection_count;

	cudaFree(state);
	cudaFree(neuron_weights);
	cudaFree(neuron_gradients_starts);
	cudaFree(connection_associated_gradient_counts);
	cudaDeviceSynchronize();

	state = tmp_state;
	neuron_weights = tmp_neuron_weights;

	cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaMalloc(&connection_associated_gradient_counts, sizeof(size_t) * neuron_count);

	cudaDeviceSynchronize();
	cudaMemcpy(neuron_gradients_starts, tmp_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_associated_gradient_counts, tmp_connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	delete[] tmp_neuron_gradients_starts;
	delete[] tmp_connection_associated_gradient_counts;
}

void LSTMLayer::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
{
	auto added_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_added_neuron(added_neuron_i, connection_probability, &added_connections_neuron_i);
	for (size_t i = 0; i < added_connections_neuron_i.size(); i++)
	{
		size_t added_neuron_i = added_connections_neuron_i[i];
		layer_gradient_count++;
		connection_associated_gradient_counts[added_neuron_i]++;
		for (size_t j = added_neuron_i + 1; j < neuron_count; j++)
			neuron_gradients_starts[j]++;
	}
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