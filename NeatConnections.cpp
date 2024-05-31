#include "NeatConnections.h"

void NeatConnections::linear_function(
	size_t activations_start, data_t* activations, 
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron
)
{
	size_t connections_start = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count = connection_counts[i];
		dim3 gridDim = dim3(connection_count / 32 + (connection_count % 32 > 0));
		cud_NEAT_neuron_linear_function kernel(gridDim, 32) (
			i, connection_count, weights, connection_points, connections_start,
			activations_start, activations,
			execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
		);
		connections_start += connection_count;
	}
	cud_add_biases kernel(dim3(neuron_count / 32 + (neuron_count % 32 > 0), 1, 1), 32) (
		neuron_count, biases, 
		execution_values_start, execution_values_layer_start, layer_execution_values_per_neuron, execution_values
	);
	cudaDeviceSynchronize();
}

void NeatConnections::calculate_derivative(
	size_t activations_start, data_t* activations, 
	size_t derivatives_start, size_t derivatives_layer_start, size_t derivatives_per_neuron, data_t* derivatives
)
{
	size_t connections_start = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count = connection_counts[i];
		cud_NEAT_linear_function_derivative kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
			activations_start, activations,
			derivatives_start, derivatives_layer_start, derivatives_per_neuron, derivatives,
			i, connection_count, weights, connection_points, connections_start
		);

		connections_start += connection_count;
	}
	cud_add_bias_derivative kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		neuron_count, derivatives_start, derivatives_layer_start, derivatives_per_neuron, derivatives
	);
	cudaDeviceSynchronize();
}

void NeatConnections::calculate_gradients(
	data_t* activations, size_t activations_start, 
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, 
	data_t* costs, size_t costs_start
)
{
	size_t connections_start = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count = connection_counts[i];
		cud_NEAT_gradient_calculation kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
			activations, activations_start,
			gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
			costs, costs_start,
			i, connection_count, weights, connection_points, connections_start
		);

		connections_start += connection_count;
	}
	cudaDeviceSynchronize();
}

void NeatConnections::subtract_gradients(
	data_t* gradients, size_t gradients_start, size_t layer_gradients_start, size_t* neuron_gradients_starts, 
	data_t learning_rate, short* dropout, data_t gradient_clip
)
{
	size_t connections_start = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		size_t connection_count = connection_counts[i];
		cud_NEAT_gradient_subtraction kernel(connection_count / 32 + (connection_count % 32 + 1), 32) (
			gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
			i, connection_count, weights, connections_start,
			learning_rate, dropout, gradient_clip
		);
		connections_start += connection_count;
	}
	cudaDeviceSynchronize();
}

void NeatConnections::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
	size_t added_connection_count = 0;
	size_t* device_tmp_connection_counts = 0;
	size_t* device_tmp_connections = 0;
	auto tmp_connections = std::vector<size_t>();
	auto sampling_vector = std::vector<size_t>();
	for (size_t i = 0; i < previous_layer_length; i++)
	{
		if (rand() % 100000 / 100000.0 < previous_layer_connection_probability)
		{
			added_connection_count++;
			tmp_connections.push_back(i + previous_layer_activations_start);
			continue;
		}
		sampling_vector.push_back(i + previous_layer_activations_start);
	}
	while (tmp_connections.size() < min_connections && sampling_vector.size())
	{
		size_t i = rand() % sampling_vector.size();
		tmp_connections.push_back(sampling_vector.at(i));
		sampling_vector.erase(sampling_vector.begin() + i);
		added_connection_count++;
	}
	
	cudaMalloc(&device_tmp_connection_counts, sizeof(size_t) * (neuron_count + 1));
	cudaMalloc(&device_tmp_connections, sizeof(size_t) * (connection_count + added_connection_count));
	cudaDeviceSynchronize();

	cudaMemcpy(device_tmp_connection_counts, connection_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_tmp_connection_counts + neuron_count, &added_connection_count, sizeof(size_t), cudaMemcpyHostToDevice);

	cudaMemcpy(device_tmp_connections, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_tmp_connections + connection_count, tmp_connections.data(), sizeof(size_t) * added_connection_count);
	cudaDeviceSynchronize();

	cudaFree(connection_counts);
	cudaFree(connection_points);
	cudaDeviceSynchronize();
	
	connection_counts = device_tmp_connection_counts;
	connection_points = device_tmp_connections;
	connection_count += added_connection_count;
}

void NeatConnections::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
{
	// Transform data to a vector
	size_t *host_connection_counts = new size_t[neuron_count];
	size_t *host_connection_points = new size_t[connection_count];
	
	cudaMemcpy(host_connection_counts, connection_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_connection_points, connection_points, sizeof(size_t) * connection_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	auto vector_connection_counts = std::vector<size_t>();
	auto vector_connection_points = std::vector<size_t>();
	for (size_t i = 0; i < connection_count; i++)
		vector_connection_points.push_back(host_connection_points[i]);

	// Add connections
	size_t tmp_connection_count = 0;
	for (size_t i = 0; i < neuron_count; i++)
	{
		uint8_t is_connection_added = rand() % 100000 / 100000 < connection_probability;
		size_t old_neuron_connection_count = host_connection_counts[i];
		size_t new_neuron_connection_count = old_neuron_connection_count + is_connection_added;
		
		vector_connection_counts.push_back(new_neuron_connection_count);
		if (is_connection_added)
			vector_connection_points.insert(vector_connection_points.begin() + tmp_connection_count + old_neuron_connection_count, added_neuron_i);

		connection_count += is_connection_added;
		tmp_connection_count += new_neuron_connection_count;
	}

	// Copy data to device, free old arrays and free host arrays
	cudaFree(connection_counts);
	cudaFree(connection_points);
	delete[] host_connection_counts;
	delete[] host_connection_points;
	cudaDeviceSynchronize();

	cudaMalloc(&connection_counts, sizeof(size_t) * neuron_count);
	cudaMalloc(&connection_points, sizeof(size_t) * connection_count);
	cudaDeviceSynchronize();

	cudaMemcpy(connection_counts, vector_connection_counts.data(), sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_points, vector_connection_points.data(), sizeof(size_t) * connection_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

