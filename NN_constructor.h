#pragma once
#ifdef INCLUDE_BACKEND

#include "ILayer.h"
#include "NeuronLayer.h"
#include "LSTMLayer.h"
#include "DenseConnections.h"
#include "NeatConnections.h"

#endif

#include "NN.h"
#include <vector>

class NN_constructor
{
private:
	size_t layer_count = 0;
	std::vector<NeuronTypes> neuron_types;
	std::vector<ConnectionTypes> connection_types;
	std::vector<size_t> layer_lengths;
	std::vector<ActivationFunctions> activations;
public:
	NN_constructor();

	NN_constructor append_layer(ConnectionTypes connections_type, NeuronTypes neurons_type, size_t neuron_count, ActivationFunctions activation = ActivationFunctions::activations_last_entry);
	NN construct(size_t input_length, bool stateful = false);
};
