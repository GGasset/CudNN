#ifndef ILAYER_DEFINITIONS
#define ILAYER_DEFINITIONS

#include "ILayer.h"

void ILayer::add_neuron(size_t neurons_to_add, size_t connection_count_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability)
{
}


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

void ILayer::delete_memory()
{
}

#endif