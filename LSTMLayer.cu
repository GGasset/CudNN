#include "LSTMLayer.h"

LSTMLayer::LSTMLayer(IConnections* connections, size_t neuron_count)
{
	layer_type = NeuronTypes::LSTM;

	is_recurrent = true;
	this->connections = connections;
	set_neuron_count(neuron_count);

	execution_values_per_neuron = 10;
	
	//derivatives_per_neuron = 19;
	derivatives_per_neuron = 24;
	layer_derivative_count = derivatives_per_neuron * neuron_count;
	
	gradients_per_neuron = 6;

	layer_gradient_count = gradients_per_neuron * neuron_count + neuron_count + connections->connection_count;

	layer_specific_initialize_fields(connections->connection_count, neuron_count);

	size_t* neuron_gradients_starts = new size_t[neuron_count];
	size_t* connection_associated_gradient_counts = new size_t[neuron_count];
	size_t gradient_count = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t neuron_connection_count = connections->get_connection_count_at(i);
		connection_associated_gradient_counts[i] = neuron_connection_count + 1;
		neuron_gradients_starts[i] = gradient_count;
		gradient_count += neuron_connection_count + 1 + gradients_per_neuron;
	}

	cudaMalloc(&this->neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaMalloc(&this->connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();

	cudaMemcpy(this->neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(this->connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	delete[] neuron_gradients_starts;
	delete[] connection_associated_gradient_counts;
}

LSTMLayer::LSTMLayer()
{
	layer_type = NeuronTypes::LSTM;
}

void LSTMLayer::layer_specific_initialize_fields(size_t connection_count, size_t neuron_count)
{
	size_t neuron_weights_count = sizeof(data_t) * neuron_count * 4;
	cudaMalloc(&state, sizeof(data_t) * neuron_count * 2);
	cudaMalloc(&neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaMalloc(&prev_state_derivatives, sizeof(data_t) * neuron_count * 3);
	cudaDeviceSynchronize();
	cudaMemset(state, 0, sizeof(data_t) * neuron_count * 2);
	cudaMemset(prev_state_derivatives, 0, sizeof(data_t) * neuron_count * 3);
	cudaMemset(neuron_weights, 0, sizeof(field_t) * neuron_count * 4);
	cudaDeviceSynchronize();
	//IConnections::generate_random_values(&neuron_weights, neuron_count * 4);
	
	add_to_array kernel(neuron_weights_count / 32 + (neuron_weights_count % 32 > 0), 32) (
		neuron_weights, neuron_weights_count, 1
	);
	cudaDeviceSynchronize();
}

ILayer* LSTMLayer::layer_specific_clone()
{
	LSTMLayer* layer = new LSTMLayer();
	cudaMalloc(&layer->neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaMalloc(&layer->state, sizeof(data_t) * neuron_count * 2);
	cudaMalloc(&layer->prev_state_derivatives, sizeof(data_t) * neuron_count * 3);
	cudaDeviceSynchronize();
	cudaMemcpy(layer->neuron_weights, neuron_weights, sizeof(field_t) * neuron_count * 4, cudaMemcpyDeviceToDevice);
	cudaMemcpy(layer->state, state, sizeof(data_t) * neuron_count * 2, cudaMemcpyDeviceToDevice);
	cudaMemcpy(layer->prev_state_derivatives, prev_state_derivatives, sizeof(data_t) * neuron_count * 3, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	return layer;
}

void LSTMLayer::specific_save(FILE* file)
{
	field_t* host_neuron_weights = new field_t[neuron_count * 4];
	data_t* host_state = new data_t[neuron_count * 2];
	data_t* host_prev_derivatives = new data_t[neuron_count * 3];

	cudaMemcpy(host_neuron_weights, neuron_weights, sizeof(neuron_count) * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_state, state, sizeof(neuron_count) * 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_prev_derivatives, prev_state_derivatives, sizeof(neuron_count) * 3, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	fwrite(host_neuron_weights, sizeof(field_t), neuron_count * 4, file);
	fwrite(host_state, sizeof(data_t), neuron_count * 2, file);
	fwrite(host_prev_derivatives, sizeof(data_t), neuron_count * 3, file);

	delete[] host_neuron_weights;
	delete[] host_state;
}

void LSTMLayer::load(FILE* file)
{
	field_t* host_neuron_weights = new field_t[neuron_count * 4];
	data_t* host_state = new data_t[neuron_count * 2];
	data_t* host_prev_derivatives = new data_t[neuron_count * 3];

	fread(host_neuron_weights, sizeof(field_t), neuron_count * 4, file);
	fread(host_state, sizeof(data_t), neuron_count * 2, file);
	fread(host_prev_derivatives, sizeof(data_t), neuron_count * 3, file);

	cudaMalloc(&neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaMalloc(&state, sizeof(data_t) * neuron_count * 2);
	cudaMalloc(&prev_state_derivatives, sizeof(data_t) * neuron_count * 3);
	cudaDeviceSynchronize();

	cudaMemcpy(neuron_weights, host_neuron_weights, sizeof(field_t) * neuron_count * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(state, host_state, sizeof(data_t) * neuron_count * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(prev_state_derivatives, host_prev_derivatives, sizeof(data_t) * neuron_count * 3, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	delete[] host_neuron_weights;
	delete[] host_state;
	delete[] host_prev_derivatives;
}

void LSTMLayer::execute(data_t* activations, size_t activations_start, data_t* execution_values, size_t execution_values_start)
{
	// neuron execution values 0
	connections->linear_function(
		activations_start, activations,
		execution_values, execution_values_start,
		execution_values_layer_start, execution_values_per_neuron
	);
	cudaDeviceSynchronize();
	LSTM_execution kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		activations, activations_start, layer_activations_start,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights, state,
		neuron_count
	);
	cudaDeviceSynchronize();

	size_t state_len = neuron_count * 2;
	reset_NaNs kernel(state_len / 32 + (state_len % 32 > 0), 32) (
		state, 0, state_len
	);
	cudaDeviceSynchronize();
}

void LSTMLayer::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* derivatives, size_t derivatives_start,
	data_t* gradients, size_t next_gradients_start, size_t gradients_start,
	data_t* costs, size_t costs_start
)
{
	LSTM_gradient_calculation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		derivatives, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
		gradients, gradients_start, next_gradients_start, layer_gradients_start, neuron_gradients_starts, connection_associated_gradient_counts,
		costs, costs_start, layer_activations_start,
		neuron_count
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
	LSTM_gradient_subtraction kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts, connection_associated_gradient_counts,
		neuron_weights, learning_rate, dropout, gradient_clip,
		neuron_count
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
	LSTM_derivative_calculation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		prev_state_derivatives, derivatives, previous_derivatives_start, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights,
		neuron_count
	);
	cudaDeviceSynchronize();
}

void LSTMLayer::mutate_fields(evolution_metadata evolution_values)
{
	float* arr = 0;
	cudaMalloc(&arr, sizeof(field_t) * neuron_count * 4 * 3);
	cudaDeviceSynchronize();
	IConnections::generate_random_values(&arr, neuron_count * 4 * 3, 0, 1);
	cudaDeviceSynchronize();

	mutate_field_array kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		neuron_weights, neuron_count, 
		evolution_values.field_mutation_chance, evolution_values.field_max_evolution, 
		arr
	);
	cudaDeviceSynchronize();
}

void LSTMLayer::layer_specific_add_neuron()
{
	field_t* tmp_neuron_weights = 0;
	data_t* tmp_state = 0;
	
	cudaMalloc(&tmp_neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaMalloc(&tmp_state, sizeof(data_t) * neuron_count * 2);
	cudaDeviceSynchronize();

	cudaMemcpy(tmp_neuron_weights, neuron_weights, sizeof(field_t) * (neuron_count - 1) * 4, cudaMemcpyDeviceToDevice);
	IConnections::generate_random_values(&tmp_neuron_weights, 4, (neuron_count - 1) * 4, 1);
	
	cudaMemcpy(tmp_state, state, sizeof(data_t) * (neuron_count - 1) * 2, cudaMemcpyDeviceToDevice);
	cudaMemset(tmp_state + (neuron_count - 1) * 4, 0, sizeof(data_t) * 4);

	cudaFree(state);
	cudaFree(neuron_weights);
	cudaDeviceSynchronize();

	state = tmp_state;
	neuron_weights = tmp_neuron_weights;
}

void LSTMLayer::remove_neuron(size_t layer_neuron_i)
{
	size_t removed_connection_count = connections->connection_count;
	connections->remove_neuron(layer_neuron_i);
	removed_connection_count -= connections->connection_count;

	set_neuron_count(neuron_count - 1);
	layer_gradient_count -= removed_connection_count + 7 + 1;
	layer_derivative_count -= derivatives_per_neuron;

	size_t *tmp_neuron_gradients_starts = neuron_gradients_starts;
	size_t *tmp_connection_associated_gradient_counts = connection_associated_gradient_counts;

	cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaMalloc(&connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();

	cudaMemcpy(neuron_gradients_starts, tmp_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(connection_associated_gradient_counts, tmp_connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	cudaFree(tmp_connection_associated_gradient_counts);
	cudaFree(tmp_neuron_gradients_starts);
	cudaDeviceSynchronize();
}

void LSTMLayer::delete_memory()
{
	cudaMemset(state, 0, sizeof(data_t) * 2 * neuron_count);
	cudaMemset(prev_state_derivatives, 0, sizeof(data_t) * 3 * neuron_count);
	cudaDeviceSynchronize();
}

void LSTMLayer::layer_specific_deallocate()
{
	cudaFree(prev_state_derivatives);
	prev_state_derivatives = 0;
	cudaFree(neuron_weights);
	neuron_weights = 0;
	cudaFree(state);
	state = 0;
}
