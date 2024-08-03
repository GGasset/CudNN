#include "NN_constructor_client.h"

NN_constructor_client NN_constructor_client::append_layer(NeuronTypes neurons, ConnectionTypes connections, int neuron_count, ActivationFunctions activations)
{
	layer_count++;
	neuron_types.push_back(neurons);
	connection_types.push_back(connections);
	layer_lengths.push_back(neuron_count);
	this->activations.push_back(activations);
}
