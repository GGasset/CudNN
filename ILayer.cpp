#ifndef ILAYER_DEFINITIONS
#define ILAYER_DEFINITIONS

#include "ILayer.h"

void ILayer::add_neuron(size_t neurons_to_add, size_t connection_count_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability = 1)
{
	parameter_t* tmp;
	tmp = weights;
	size_t new_weight_count = connection_count + sizeof(parameter_t) * neurons_to_add * connection_count_per_neuron;
	cudaMalloc(&weights, new_weight_count);
	cudaMemcpy(weights, tmp, new_weight_count, cudaMemcpyDeviceToDevice);
	cudaFree(tmp);
	connections->add_neuron(neurons_to_add, connection_count_per_neuron, layer_i, layer_i_prev_length, connection_probability);
	connection_count += neurons_to_add * connection_count_per_neuron;
	neuron_count += neurons_to_add;
}

void ILayer::generate_random_weights(size_t connection_count, size_t start_i = 0)
{
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	curandGenerateUniform(generator, weights + start_i, connection_count);
}


void ILayer::initialize_fields(size_t connection_count, size_t neuron_count)
{
	cudaMalloc(&weights, sizeof(parameter_t) * connection_count);
	generate_random_weights(connection_count);
	cudaMalloc(&biases, sizeof(parameter_t) * neuron_count);
	cudaMemset(biases, 1, sizeof(parameter_t) * neuron_count);
}

void ILayer::deallocate()
{
	cudaFree(weights);
	cudaFree(biases);
	connections->deallocate();
	cudaDeviceSynchronize();
	delete connections;
}

#endif