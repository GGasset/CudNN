#ifndef ILAYER_DEFINITIONS
#define ILAYER_DEFINITIONS

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

	base_layer->set_neuron_count(get_neuron_count);

	base_layer->execution_values_per_neuron = execution_values_per_neuron;
	
	base_layer->layer_derivative_count = layer_derivative_count;
	base_layer->derivatives_per_neuron = derivatives_per_neuron;

	base_layer->layer_gradient_count = layer_gradient_count;
	
	cudaMalloc(&base_layer->neuron_gradients_starts, sizeof(size_t) * get_neuron_count());
	if (connection_associated_gradient_counts)
		cudaMalloc(&base_layer->connection_associated_gradient_counts, sizeof(size_t) * get_neuron_count());
	cudaDeviceSynchronize();

	cudaMemcpy(base_layer->neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * get_neuron_count(), cudaMemcpyDeviceToDevice);
	if (connection_associated_gradient_count)
		cudaMemcpy(base_layer->connection_associated_gradient_counts, connection_associated_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToDevice);
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

void ILayer::delete_memory()
{
}

#endif
