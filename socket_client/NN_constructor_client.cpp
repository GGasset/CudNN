#include "NN_constructor_client.h"

NN_constructor_client* NN_constructor_client::append_layer(NeuronTypes neurons, ConnectionTypes connections, int neuron_count, ActivationFunctions activations)
{
	layer_count++;
	neuron_types.push_back(neurons);
	connection_types.push_back(connections);
	layer_lengths.push_back(neuron_count);
	this->activations.push_back(activations);
	return this;
}


NN_constructor_client* NN_constructor_client::create_minimal_recurrent_NN(int input_length, bool stateful, int output_length)
{
	set_fields(input_length, stateful);
	append_layer(NeuronTypes::LSTM, ConnectionTypes::NEAT, output_length, ActivationFunctions::activations_last_entry);
	return this;
}
