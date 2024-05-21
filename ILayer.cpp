#ifndef ILAYER_DEFINITIONS
#define ILAYER_DEFINITIONS

#include "ILayer.h"

void ILayer::add_neuron(size_t neurons_to_add, size_t connection_count_per_neuron, size_t layer_i, size_t layer_i_prev_length, float connection_probability)
{
	field_t* tmp;
	tmp = weights;
	size_t new_weight_count = connection_count + sizeof(field_t) * neurons_to_add * connection_count_per_neuron;
	cudaMalloc(&weights, new_weight_count);
	cudaMemcpy(weights, tmp, new_weight_count, cudaMemcpyDeviceToDevice);
	cudaFree(tmp);
	connections->add_neuron(neurons_to_add, connection_count_per_neuron, layer_i, layer_i_prev_length, connection_probability);
	connection_count += neurons_to_add * connection_count_per_neuron;
	neuron_count += neurons_to_add;
}

void ILayer::generate_random_values(float** pointer, size_t float_count, size_t start_i)
{
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	//curandSetPseudoRandomGeneratorSeed(generator, 15);
	curandGenerateUniform(generator, *pointer + start_i, float_count);
}


void ILayer::initialize_fields(size_t connection_count, size_t neuron_count)
{
	cudaMalloc(&weights, sizeof(field_t) * connection_count);
	cudaMalloc(&biases, sizeof(field_t) * neuron_count);
	cudaDeviceSynchronize();
	generate_random_values(&weights, connection_count);
	generate_random_values(&biases, connection_count);
	layer_specific_initialize_fields(connection_count, neuron_count);
	cudaDeviceSynchronize();
}

void ILayer::layer_specific_initialize_fields(size_t connection_count, size_t neuron_count)
{
}

void ILayer::deallocate()
{
	cudaFree(weights);
	cudaFree(biases);
	connections->deallocate();
	layer_specific_deallocate();
	cudaDeviceSynchronize();
	delete connections;
}

void ILayer::layer_specific_deallocate()
{

}

void ILayer::delete_memory()
{
}

#endif