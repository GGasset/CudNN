#ifndef NEURONLAYER_DEFINITIONS
#define NEURONLAYER_DEFINITIONS

#include "NeuronLayer.h"

void NeuronLayer::execute(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start
)
{
	connections->linear_function(activations_start, activations,
		execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron
	);
	switch (activation)
	{
	case ActivationFunctions::sigmoid:
		sigmoid_activation kernel(1, neuron_count) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0
		);
		break;
	case ActivationFunctions::_tanh:
		tanh_activation kernel(1, neuron_count) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0
		);
		break;
	default:
		break;
	}
	cudaDeviceSynchronize();
}

void NeuronLayer::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* derivatives, size_t derivatives_start,
	data_t* gradients, size_t next_gradients_start, size_t gradients_start,
	data_t* costs, size_t costs_start
)
{
	neuron_gradient_calculation kernel(1, neuron_count) (
		execution_values, execution_values_start, execution_values_layer_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start, layer_activations_start,
		activation
	);
	cudaDeviceSynchronize();
	connections->calculate_gradients(
		activations, activations_start, gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start
	);
	cudaDeviceSynchronize();
}

void NeuronLayer::subtract_gradients(data_t* gradients, size_t gradients_start, data_t learning_rate, short* dropout, data_t gradient_clip)
{
	connections->subtract_gradients(
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		learning_rate, dropout, gradient_clip
	);
}

void NeuronLayer::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
	
	connections->add_neuron(previous_layer_length, previous_layer_activations_start, previous_layer_connection_probability, min_connections);
	set_neuron_count(neuron_count + 1);
}

#endif