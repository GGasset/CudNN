#pragma once
#include "LSTMLayer.h"
#include "DenseConnections.h"
class DenseLSTMLayer :
    public LSTMLayer
{
public:
    DenseLSTMLayer(size_t neuron_count, size_t previous_layer_neuron_i_start, size_t previous_layer_length);
};

