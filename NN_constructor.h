#pragma once
#ifndef HEADER_ONLY

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
	std::vector<NN::NeuronTypes> neuron_types;
	std::vector<NN::ConnectionTypes> connection_types;
	std::vector<size_t> layer_lengths;
	std::vector<ActivationFunctions> activations;
public:
	NN_constructor();

	NN_constructor append_layer(NN::ConnectionTypes connections_type, NN::NeuronTypes neurons_type, size_t neuron_count, ActivationFunctions activation = ActivationFunctions::activations_last_entry);
	NN construct(size_t input_length, bool stateful = false);
};
