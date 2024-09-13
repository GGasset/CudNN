#ifndef NEURONLAYER_DEFINITIONS
#define NEURONLAYER_DEFINITIONS

#include "NeuronLayer.h"
#include <stdio.h>

NeuronLayer::NeuronLayer(IConnections* connections, size_t neuron_count, ActivationFunctions activation)
{
	this->connections = connections;
	set_neuron_count(neuron_count);
	this->activation = activation;
	execution_values_per_neuron = 1;
	layer_gradient_count = connections->connection_count + neuron_count;

	size_t neuron_gradient_i = 0;
	size_t* host_neuron_gradients_starts = new size_t[neuron_count];
	for (size_t i = 0; i < neuron_count; i++)
	{
		host_neuron_gradients_starts[i] = neuron_gradient_i;

		size_t neuron_connection_count = connections->get_connection_count_at(i);
		neuron_gradient_i += neuron_connection_count + 1;
	}

	cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();
	
	cudaMemcpy(neuron_gradients_starts, host_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

NeuronLayer::NeuronLayer()
{
	
}

void NeuronLayer::layer_specific_deallocate()
{
	cudaFree(neuron_gradients_starts);
	if (connection_associated_gradient_counts)
		cudaFree(connection_associated_gradient_counts);
}

ILayer* NeuronLayer::layer_specific_clone()
{
	NeuronLayer* layer = new NeuronLayer();
	layer->activation = activation;
	return layer;
}

void NeuronLayer::specific_save(FILE* file)
{
	size_t activation_function = (size_t)activation;
	fwrite(&activation_function, sizeof(size_t), 1, file);
}

void NeuronLayer::load(FILE* file)
{
	ILayer_load(file);
	
	size_t activation_function = 0;
	fread(&activation_function, sizeof(size_t), 1, file);
	activation = (ActivationFunctions)activation_function;
}

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
		sigmoid_activation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0,
			neuron_count
		);
		break;
	case ActivationFunctions::_tanh:
		tanh_activation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
			activations, activations_start, layer_activations_start, true,
			execution_values, execution_values_start, execution_values_layer_start, execution_values_per_neuron, 0, 0, 0,
			neuron_count
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
	neuron_gradient_calculation kernel(neuron_count / 32 + (neuron_count % 32 > 0), 32) (
		execution_values, execution_values_start, execution_values_layer_start,
		gradients, gradients_start, layer_gradients_start, neuron_gradients_starts,
		costs, costs_start, layer_activations_start,
		activation,
		neuron_count
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
	size_t added_connections = connections->connection_count;
	connections->add_neuron(previous_layer_length, previous_layer_activations_start, previous_layer_connection_probability, min_connections);
	added_connections = connections->connection_count - added_connections;


	layer_derivative_count += derivatives_per_neuron;
	layer_gradient_count++;
	layer_gradient_count += added_connections;
	
	size_t* new_neuron_gradients_starts = new size_t[neuron_count + 1];
	cudaMemcpy(new_neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyDeviceToHost);
	new_neuron_gradients_starts[neuron_count] = new_neuron_gradients_starts[neuron_count - 1] + added_connections;
	cudaDeviceSynchronize();
	
	cudaFree(neuron_gradients_starts);
	cudaDeviceSynchronize();

	set_neuron_count(neuron_count + 1);

	cudaMalloc(&neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();
	cudaMemcpy(neuron_gradients_starts, new_neuron_gradients_starts, sizeof(size_t) * neuron_count, cudaMemcpyHostToDevice);
	delete[] new_neuron_gradients_starts;
	cudaDeviceSynchronize();
}

void NeuronLayer::adjust_to_added_neuron(size_t added_neuron_i, float connection_probability)
{
	auto added_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_added_neuron(added_neuron_i, connection_probability, &added_connections_neuron_i);
	for (size_t i = 0; i < added_connections_neuron_i.size(); i++)
	{
		layer_gradient_count++;
		size_t added_connection_neuron_i = added_connections_neuron_i[i];
		size_t to_modify_neuron_count = neuron_count - added_connection_neuron_i - 1;
		if (to_modify_neuron_count)
			add_to_array kernel(to_modify_neuron_count / 32 + (to_modify_neuron_count % 32 > 0), 32) (
				neuron_gradients_starts + added_connection_neuron_i + 1, to_modify_neuron_count, 1
			);
	}
}

void NeuronLayer::remove_neuron(size_t layer_neuron_i)
{
	size_t removed_connection_count = connections->connection_count;
	size_t* tmp_neuron_gradients_starts = 0;
	
	connections->remove_neuron(layer_neuron_i);
	removed_connection_count -= connections->connection_count;
	layer_gradient_count -= removed_connection_count;

	set_neuron_count(neuron_count - 1);

	cudaMalloc(&tmp_neuron_gradients_starts, sizeof(size_t) * neuron_count);
	cudaDeviceSynchronize();
	cudaMemcpy(tmp_neuron_gradients_starts, neuron_gradients_starts, sizeof(size_t) * layer_neuron_i, cudaMemcpyDeviceToDevice);
	cudaMemcpy(tmp_neuron_gradients_starts + layer_neuron_i, neuron_gradients_starts + layer_neuron_i + 1, sizeof(size_t) * (neuron_count - layer_neuron_i), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cudaFree(neuron_gradients_starts);
	cudaDeviceSynchronize();
	neuron_gradients_starts = tmp_neuron_gradients_starts;
}

void NeuronLayer::adjust_to_removed_neuron(size_t neuron_i)
{
	auto deleted_connections_neuron_i = std::vector<size_t>();
	connections->adjust_to_removed_neuron(neuron_i, &deleted_connections_neuron_i);
	for (size_t i = 0; i < deleted_connections_neuron_i.size(); i++)
	{
		layer_gradient_count--;
		size_t removed_connection_neuron_i = deleted_connections_neuron_i[i];
		size_t to_modify_neuron_count = neuron_count - removed_connection_neuron_i - 1;
		if (to_modify_neuron_count)
			add_to_array kernel(to_modify_neuron_count / 32 + (to_modify_neuron_count % 32 > 0), 32) (
				neuron_gradients_starts + removed_connection_neuron_i + 1, to_modify_neuron_count, -1
			);
	}
}

#endif
