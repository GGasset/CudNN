#include "IConnections.h"

void IConnections::generate_random_values(float** pointer, size_t float_count, size_t start_i)
{
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	//curandSetPseudoRandomGeneratorSeed(generator, 15);
	curandGenerateUniform(generator, *pointer + start_i, float_count);
}

void IConnections::add_neuron(size_t previous_layer_length, size_t previous_layer_activations_start, float previous_layer_connection_probability, size_t min_connections)
{
}

void IConnections::remove_neuron(size_t neuron_i)
{
}

void IConnections::adjust_to_removed_neuron(size_t neuron_i, std::vector<size_t>* removed_connections_neuron_i)
{
}

void IConnections::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability, std::vector<size_t>* added_connections_neuron_i)
{
}

void IConnections::deallocate()
{
	cudaFree(weights);
	cudaFree(biases);
	specific_deallocate();
	cudaDeviceSynchronize();
}

void IConnections::specific_deallocate()
{
}
