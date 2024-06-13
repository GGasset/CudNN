#include "IConnections.h"

void IConnections::generate_random_values(float** pointer, size_t float_count, size_t start_i)
{
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	//curandSetPseudoRandomGeneratorSeed(generator, 15);
	curandGenerateUniform(generator, *pointer + start_i, float_count);
}

void IConnections::mutate_fields(evolution_metadata evolution_values)
{
	float* arr0 = 0;
	cudaMalloc(&arr0, sizeof(float) * neuron_count * 3);
	cudaDeviceSynchronize();
	generate_random_values(&arr0, neuron_count * 3);
	cudaDeviceSynchronize();

	mutate_field_array kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		biases, neuron_count,
		evolution_values.field_max_evolution, evolution_values.field_mutation_chance,
		arr0
	);
	cudaFree(arr0);
	cudaDeviceSynchronize();

	cudaMalloc(&arr0, sizeof(float) * connection_count * 3);
	cudaDeviceSynchronize();
	generate_random_values(&arr0, connection_count * 3);
	cudaDeviceSynchronize();

	mutate_field_array kernel(connection_count / 32 + (connection_count % 32 > 0), 32) (
		weights, connection_count,
		evolution_values.field_max_evolution, evolution_values.field_mutation_chance,
		arr0
	);

	cudaFree(arr0);
	cudaDeviceSynchronize();
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
