#ifndef LSTMLAYER_DEFINITIONS
#define LSTMLAYER_DEFINITIONS

#include "LSMTLayer.h"



#endif

void LSMTLayer::execute(data_t* activations, size_t activations_start, data_t* execution_values, size_t execution_values_start)
{
	// neuron execution values 0
	connections->linear_function(
		activations_start, activations,
		execution_values, execution_values_start,
		execution_values_layer_start, execution_values_per_neuron,
		weights, biases, neuron_count
	);


}