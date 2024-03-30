#include "NN.h"

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