#include "NeuronLayer.h"
#include "DenseConnections.h"

#pragma once
class DenseNeuronLayer : public NeuronLayer
{
public:
	DenseNeuronLayer(size_t layer_gradients_start, size_t neuron_count, size_t previous_layer_neuron_i_start, size_t previous_layer_length, ActivationFunctions activation);
};

