#include "DenseNeuronLayer.h"

DenseNeuronLayer::DenseNeuronLayer(size_t neuron_count, size_t previous_layer_neuron_i_start, size_t previous_layer_length, ActivationFunctions activation)
{
	connections = new DenseConnections(previous_layer_neuron_i_start, previous_layer_length);
	this->activation = activation;
	this->neuron_count = neuron_count;
	this->connection_count = neuron_count * previous_layer_length;
	execution_values_per_neuron = 1;
	layer_gradient_count = connection_count + neuron_count;
	initialize_fields(previous_layer_length * neuron_count, neuron_count);
}