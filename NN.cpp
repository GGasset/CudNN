#ifndef NN_DEFINITIONS
#define NN_DEFINITIONS

#include "NN.h"

NN::NN(ILayer** layers, size_t input_length, size_t layer_count, size_t max_layer_count)
{
	// set max layer count to layer count if max_layer_count is lesser than layer count
	max_layer_count += (layer_count - max_layer_count) * (max_layer_count < layer_count);
	
	this->layers = layers;
	this->input_length = input_length;
	this->layer_count = layer_count;
	this->max_layer_count = layer_count;
	set_fields();
}

void NN::set_fields()
{
	output_length = layers[layer_count - 1]->neuron_count;

	size_t neuron_count = input_length;
	size_t execution_value_count = 0;
	size_t gradient_count = 0;
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* layer = layers[i];

		layer->layer_activations_start = neuron_count;
		neuron_count += layer->neuron_count;

		layer->execution_values_layer_start = execution_value_count;
		execution_value_count += layer->execution_values_per_neuron * layer->neuron_count;

		gradient_count += layer->layer_gradient_count;
	}
	this->neuron_count = neuron_count;
	output_activations_start = &(layers[layer_count - 1]->layer_activations_start);
	this->execution_value_count = execution_value_count;
}

void NN::execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, short copy_output_to_host = true)
{
	cudaMemcpy(activations + t * neuron_count, input + input_length * t, sizeof(data_t) * input_length, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->execute(activations, neuron_count * t, execution_values, execution_value_count * t);
	}
	if (copy_output_to_host)
	{
		cudaMemcpy(output_start_pointer + output_length * t, activations + neuron_count * t + *output_activations_start, sizeof(data_t) * output_length, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
}

#include <stdio.h>

data_t* NN::execute(data_t* input, size_t t_count)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	cudaMalloc(&execution_values, sizeof(data_t) * execution_value_count * t_count);
	cudaMalloc(&activations, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
	cudaMemset(execution_values, 0, sizeof(data_t) * execution_value_count * t_count);
	cudaMemset(activations, 0, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();

	//data_t* host_activations = new data_t[neuron_count];
	//cudaMemcpy(host_activations, activations, sizeof(data_t) * neuron_count, cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

	//for (size_t i = 0; i < neuron_count; i++)
	//{
	//	printf("%f ", host_activations[i]);
	//}
	//printf("\n\n");

	data_t* outputs = new data_t[output_length * t_count];
	for (size_t i = 0; i < output_length * t_count; i++)
	{
		outputs[i] = 0;
	}
	for (size_t i = 0; i < t_count; i++)
		execute(input, execution_values, activations, i, outputs + output_length * i, 1);


	cudaFree(execution_values);
	cudaFree(activations);
	cudaDeviceSynchronize();
	return outputs;
}

data_t* NN::execute(data_t* input)
{
	return execute(input, 1);
}

void NN::deallocate()
{
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->deallocate();
		delete layers[i];
	}
	delete[] layers;
}

#endif