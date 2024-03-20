#include "ILayer.h"

void ILayer::add_neuron(size_t neurons_to_add, size_t connection_count_per_neuron)
{
	parameter_t* tmp;
	tmp = weights;
	size_t new_weight_count = connection_count + sizeof(parameter_t) * neurons_to_add * connection_count_per_neuron;
	cudaMalloc(&weights, new_weight_count);
	cudaMemcpy(weights, tmp, new_weight_count, cudaMemcpyDeviceToDevice);
	cudaFree(tmp);
	connection_count += neurons_to_add * connection_count_per_neuron;
	neuron_count += neurons_to_add;
}