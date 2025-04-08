#include "ILayer.h"

size_t ILayer::get_neuron_count()
{
	return neuron_count;
}

void ILayer::set_neuron_count(size_t neuron_count)
{
	this->neuron_count = neuron_count;
	connections->neuron_count = neuron_count;
}

void ILayer::initialize_fields(size_t connection_count, size_t neuron_count)
{
	layer_specific_initialize_fields(connection_count, neuron_count);
	cudaDeviceSynchronize();
}

void ILayer::layer_specific_initialize_fields(size_t connection_count, size_t neuron_count)
{
}

void ILayer::ILayerClone(ILayer* base_layer)
{
	IConnections* cloned_connections = connections->connections_specific_clone();
	connections->IConnections_clone(cloned_connections);
	base_layer->connections = cloned_connections;

	base_layer->set_neuron_count(get_neuron_count());

	base_layer->execution_values_per_neuron = execution_values_per_neuron;
	
	base_layer->layer_derivative_count = layer_derivative_count;
	base_layer->derivatives_per_neuron = derivatives_per_neuron;

	base_layer->layer_gradient_count = layer_gradient_count;
	
	cudaMalloc(&base_layer->neuron_gradients_starts, sizeof(size_t) * get_neuron_count());
	if (connection_associated_gradient_counts)
		cudaMalloc(&base_layer->connection_associated_gradient_counts, sizeof(size_t) * get_neuron_count());
	cudaDeviceSynchronize();

	cudaMemcpy(base_layer->neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * get_neuron_count(), cudaMemcpyDeviceToDevice);
	if (connection_associated_gradient_counts)
		cudaMemcpy(base_layer->connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
}

void ILayer::save(FILE* file)
{
	fwrite(&neuron_count, sizeof(size_t), 1, file);
	fwrite(&execution_values_per_neuron, sizeof(size_t), 1, file);
	fwrite(&layer_derivative_count, sizeof(size_t), 1, file);
	fwrite(&derivatives_per_neuron, sizeof(size_t), 1, file);
	fwrite(&layer_gradient_count, sizeof(size_t), 1, file);
	
	size_t *host_neuron_gradients_starts, *host_connection_gradient_counts;
	host_neuron_gradients_starts = new size_t[neuron_count];
	host_connection_gradient_counts = new size_t[neuron_count];

	cudaMemcpy(host_neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_connection_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	fwrite(host_neuron_gradients_starts, sizeof(size_t), neuron_count, file);
	fwrite(host_connection_gradient_counts, sizeof(size_t), neuron_count, file);
	delete[] host_neuron_gradients_starts;
	delete[] host_connection_gradient_counts;

	specific_save(file);
}

void ILayer::ILayer_load(FILE* file)
{
	fread(&neuron_count, sizeof(size_t), 1, file);
	fread(&execution_values_per_neuron, sizeof(size_t), 1, file);
	fread(&layer_derivative_count, sizeof(size_t), 1, file);
	fread(&derivatives_per_neuron, sizeof(size_t), 1, file);
	fread(&layer_gradient_count, sizeof(size_t), 1, file);

	size_t* host_neuron_gradients_starts = new size_t[neuron_count];
	size_t* host_connection_associated_gradient_counts = new size_t[neuron_count];

	fread(host_neuron_gradients_starts, sizeof(size_t), neuron_count, file);
	fread(host_connection_associated_gradient_counts, sizeof(size_t), neuron_count, file);

	cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaMalloc(&connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();

	cudaMemcpy(neuron_gradients_starts, host_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(connection_associated_gradient_counts, host_connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	delete[] host_neuron_gradients_starts;
	delete[] host_connection_associated_gradient_counts;
}

void ILayer::deallocate()
{
	connections->deallocate();
	layer_specific_deallocate();
	cudaDeviceSynchronize();
	delete connections;
}

void ILayer::layer_specific_deallocate()
{

}

void ILayer::mutate_fields(evolution_metadata evolution_values)
{
}

void ILayer::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
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
			if (connection_associated_gradient_counts)
				add_to_array kernel(1, 1) (
					connection_associated_gradient_counts + added_connection_neuron_i, 1, 1
				);
			if (neuron_gradients_starts)
				add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
					neuron_gradients_starts + added_connection_neuron_i + 1, remaining_neuron_count, 1
				);
		}
	}
}

void ILayer::adjust_to_removed_neuron(size_t neuron_i)
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
			if (connection_associated_gradient_counts)
				add_to_array kernel(1, 1) (
					connection_associated_gradient_counts + removed_connection_neuron_i, 1, -1
				);
			if (neuron_gradients_starts)
				add_to_array kernel(remaining_neuron_count / 32 + (remaining_neuron_count % 32 > 0), 32) (
					neuron_gradients_starts + removed_connection_neuron_i + 1, remaining_neuron_count, -1
				);
		}
	}
}

void ILayer::delete_memory()
{
}
