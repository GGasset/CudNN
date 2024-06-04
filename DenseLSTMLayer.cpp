#include "DenseLSTMLayer.h"

DenseLSTMLayer::DenseLSTMLayer(size_t layer_gradients_start, size_t neuron_count, size_t previous_layer_neuron_i_start, size_t previous_layer_length)
{
	connections = new DenseConnections(previous_layer_neuron_i_start, previous_layer_length);
	set_neuron_count(neuron_count);
	execution_values_per_neuron = 10;

	size_t* connection_gradient_counts = new size_t[neuron_count];
	size_t neuron_gradient_i = layer_gradients_start;
	size_t* neuron_gradients_starts = new size_t[neuron_count];
	for (size_t i = 0; i < neuron_count; i++)
	{
		connection_gradient_counts[i] = previous_layer_length + 1;
		neuron_gradients_starts[i] = neuron_gradient_i;
		neuron_gradient_i += 1 + previous_layer_length + 7;
	}
	cudaMalloc(&connection_associated_gradient_counts, sizeof(size_t) * neuron_count);
	cudaMalloc(&this->neuron_gradients_starts, neuron_count * sizeof(size_t));
	cudaDeviceSynchronize();
	cudaMemcpy(connection_associated_gradient_counts, connection_gradient_counts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(this->neuron_gradients_starts, neuron_gradients_starts, neuron_count * sizeof(size_t), cudaMemcpyHostToDevice);
	delete[] connection_gradient_counts;
	delete[] neuron_gradients_starts;

	derivatives_per_neuron = 16;
	layer_derivative_count = derivatives_per_neuron * neuron_count;
	layer_gradient_count = neuron_count + connections->connection_count + 7 * neuron_count;
	initialize_fields(previous_layer_length * neuron_count, neuron_count);
}