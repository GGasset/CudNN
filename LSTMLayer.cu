#include "LSTMLayer.h"

LSTMLayer::LSTMLayer(IConnections* connections, size_t neuron_count)
{
	is_recurrent = true;
	this->connections = connections;
	set_neuron_count(neuron_count);

	execution_values_per_neuron = 10;
	
	derivatives_per_neuron = 16;
	layer_derivative_count = derivatives_per_neuron * neuron_count;
	
	layer_gradient_count = 8 * neuron_count + neuron_count + connections->connection_count;

	layer_specific_initialize_fields(connections->connection_count, neuron_count);

	size_t* neuron_gradients_starts = new size_t[neuron_count];
	size_t* connection_associated_gradient_counts = new size_t[neuron_count];
	size_t gradient_count = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t neuron_connection_count = connections->get_connection_count_at(i);
		connection_associated_gradient_counts[i] = neuron_connection_count + 1;
		neuron_gradients_starts[i] = gradient_count;
		gradient_count += neuron_connection_count + 1;
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

}

void LSTMLayer::layer_specific_initialize_fields(size_t connection_count, size_t neuron_count)
{
	cudaMalloc(&state, sizeof(data_t) * neuron_count * 2);
	cudaMalloc(&neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaDeviceSynchronize();
	cudaMemset(state, 0, sizeof(data_t) * neuron_count * 2);
	IConnections::generate_random_values(&neuron_weights, neuron_count * 4);
	cudaDeviceSynchronize();
}

ILayer* LSTMLayer::layer_specific_clone()
{
	LSTMLayer* layer = new LSTMLayer();
	cudaMalloc(&layer->neuron_weights, sizeof(field_t) * neuron_count * 4);
	cudaMalloc(&layer->state, sizeof(data_t) * neuron_count * 2);
	cudaDeviceSynchronize();
	cudaMemcpy(layer->neuron_weights, neuron_weights, sizeof(field_t) * neuron_count * 4, cudaMemcpyDeviceToDevice);
	cudaMemcpy(layer->state, state, sizeof(data_t) * neuron_count * 2, cudaMemcpyDeviceToDevice);
	return layer;
}

void LSTMLayer::specific_save(FILE* file)
{
	field_t* host_neuron_weights = new field_t[neuron_count * 4];
	data_t* host_state = new data_t[neuron_count * 2];

	cudaMemcpy(host_neuron_weights, neuron_weights, sizeof(neuron_count) * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_state, state, sizeof(neuron_count) * 2, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	fwrite(host_neuron_weights, sizeof(field_t), neuron_count * 4, file);
	fwrite(host_state, sizeof(data_t), neuron_count * 2, file);

	delete[] host_neuron_weights;
	delete[] host_state;
}

void LSTMLayer::execute(data_t* activations, size_t activations_start, data_t* execution_values, size_t execution_values_start)
{
	// neuron execution values 0
	connections->linear_function(
		activations_start, activations,
		execution_values, execution_values_start,
		execution_values_layer_start, execution_values_per_neuron
	);
	LSTM_execution kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		activations, activations_start, layer_activations_start,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights, state,
		neuron_count
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
		derivatives, previous_derivatives_start, derivatives_start, layer_derivatives_start, derivatives_per_neuron,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron,
		neuron_weights,
		neuron_count
	);
}

void LSTMLayer::mutate_fields(evolution_metadata evolution_values)
{
	float* arr = 0;
	cudaMalloc(&arr, sizeof(field_t) * neuron_count * 4 * 3);
	cudaDeviceSynchronize();
	IConnections::generate_random_values(&arr, neuron_count * 4 * 3);
	cudaDeviceSynchronize();

	mutate_field_array kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		neuron_weights, neuron_count, 
		evolution_values.field_mutation_chance, evolution_values.field_max_evolution, 
		arr
	);
	cudaDeviceSynchronize();
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
	cudaMemcpy(tmp_connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count - 1, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	tmp_neuron_gradients_starts[neuron_count - 1] = tmp_neuron_gradients_starts[neuron_count - 2] + tmp_connection_associated_gradient_counts[neuron_count - 2] + 7;
	tmp_connection_associated_gradient_counts[neuron_count - 1] = added_connection_count;

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

	layer_derivative_count += derivatives_per_neuron;
	layer_gradient_count += added_connection_count + 7 + 1;
}

void LSTMLayer::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
{
	auto added_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_added_neuron(added_neuron_i, connection_probability, &added_connections_neuron_i);
	for (size_t i = 0; i < added_connections_neuron_i.size(); i++)
	{
		layer_gradient_count++;
		size_t added_connection_neuron_i = added_connections_neuron_i[i];
		size_t remaining_neuron_count = neuron_count - added_connection_neuron_i - 1;
		if (remaining_neuron_count)
		{
			add_to_array kernel(1, 1) (
				connection_associated_gradient_counts + added_connection_neuron_i, 1, 1
			);
			add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
				neuron_gradients_starts + added_connection_neuron_i + 1, remaining_neuron_count, 1
			);
		}
	}
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

void LSTMLayer::adjust_to_removed_neuron(size_t neuron_i)
{
	auto removed_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_removed_neuron(neuron_i, &removed_connections_neuron_i);
	for (size_t i = 0; i < removed_connections_neuron_i.size(); i++)
	{
		layer_gradient_count--;
		size_t removed_connection_neuron_i = removed_connections_neuron_i[i];
		size_t remaining_neuron_count = neuron_count - removed_connection_neuron_i - 1;
		if (remaining_neuron_count)
		{
			add_to_array kernel(1, 1) (
				connection_associated_gradient_counts + removed_connection_neuron_i, 1, -1
			);
			add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
				neuron_gradients_starts + removed_connection_neuron_i + 1, remaining_neuron_count, -1
			);
		}
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
