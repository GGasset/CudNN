#include "IConnections.h"

#pragma once
class ILayer
{
	IConnections* connections = 0;
	parameter_t* weights = 0;
	size_t* weights_starts = 0;
	parameter_t* biases = 0;
	size_t execution_values_layer_start = 0;
	size_t execution_values_per_neuron = 0;

};

