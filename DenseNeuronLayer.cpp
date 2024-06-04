#ifndef DENSE_NEURON_CONSTRUCTOR
#define DENSE_NEURON_CONSTRUCTOR

#include "DenseNeuronLayer.h"

DenseNeuronLayer::DenseNeuronLayer(size_t neuron_count, size_t previous_layer_neuron_i_start, size_t previous_layer_length, ActivationFunctions activation)
{
	connections = new DenseConnections(previous_layer_neuron_i_start, previous_layer_length, neuron_count);
	set_neuron_count(neuron_count);
	this->activation = activation;
	execution_values_per_neuron = 1;
	
	size_t neuron_gradient_i = layer_gradients_start;
	size_t* neuron_gradients_starts = new size_t[neuron_count];
	for (size_t i = 0; i < neuron_count; i++)
	{
		neuron_gradients_starts[i] = neuron_gradient_i;
		neuron_gradient_i += previous_layer_length + 1;
	}
	cudaMalloc(&this->neuron_gradients_starts, neuron_count * sizeof(size_t));
	cudaDeviceSynchronize();
	cudaMemcpy(this->neuron_gradients_starts, neuron_gradients_starts, neuron_count * sizeof(size_t), cudaMemcpyHostToDevice);
	delete[] neuron_gradients_starts;

	layer_gradient_count = connections->connection_count + neuron_count;
	initialize_fields(previous_layer_length * neuron_count, neuron_count);
}

#endif