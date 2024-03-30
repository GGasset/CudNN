#include "NN.h"

void NN::set_fields()
{
	size_t neuron_count = 0;
	size_t execution_value_count = 0;
	size_t gradient_count = 0;
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* layer = layers[i];

		layer->layer_activations_start = neuron_count;
		neuron_count += layer->neuron_count;

		layer->execution_values_layer_start = execution_value_count;
		execution_value_count += layer->execution_values_per_neuron * layer->neuron_count;

		layer->gradients_start = gradient_count;
		gradient_count += layer->layer_gradient_count;
	}
	this->neuron_count = neuron_count;
	output_activations_start = &(layers[layer_count - 1]->layer_activations_start);
}

void NN::Execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, short copy_output_to_host = true)
{
	cudaMemcpy(activations + t * neuron_count, input + input_length * t, input_length, cudaMemcpyHostToDevice);
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->execute(activations, neuron_count * t, execution_values, execution_value_count * t);
	}
	if (copy_output_to_host)
	{
		cudaMemcpy(output_start_pointer + output_length * t, activations + neuron_count * t + *output_activations_start, output_length, cudaMemcpyDeviceToHost);
	}
}

data_t* NN::Execute(data_t* input, size_t t_count)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	cudaMalloc(&execution_values, execution_value_count * t_count);
	cudaMalloc(&activations, neuron_count * t_count);

	data_t* outputs = new data_t[output_length * t_count];
	for (size_t i = 0; i < t_count; i++)
	{
		Execute(input, execution_values, activations, i, outputs + output_length * i, 1);
	}
	cudaFree(execution_values);
	cudaFree(activations);
	return outputs;
}

data_t* NN::Execute(data_t* input)
{
	return Execute(input, 1);
}