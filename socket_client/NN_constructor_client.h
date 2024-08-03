#include <vector>

#include "NN_enums.h"

#pragma once
class NN_constructor_client
{
public:
	size_t input_length = 0;
	bool stateful = false;
	size_t layer_count = 0;
	std::vector<NeuronTypes> neuron_types;
	std::vector<ConnectionTypes> connection_types;
	std::vector<int> layer_lengths;
	std::vector<ActivationFunctions> activations;

	NN_constructor_client(int input_length, bool stateful)
	{
		this->input_length = input_length;
		this->stateful = stateful;
	}

	NN_constructor_client append_layer(NeuronTypes neurons, ConnectionTypes connections, int neuron_count, ActivationFunctions activations);
};
