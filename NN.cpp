#include "NN.h"

data_t* NN::Execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, short copy_output_to_host = true)
{
	cudaMemcpy(activations + t * neuron_count, input + input_length * t, input_length, cudaMemcpyHostToDevice);
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->execute(activations, neuron_count * t, execution_values, execution_value_count * t);
	}
	data_t* output = 0;
	if (copy_output_to_host)
	{
		output = new data_t[output_length];
		cudaMemcpy(output, activations + neuron_count * t + *output_activations_start, output_length, cudaMemcpyDeviceToHost);
	}
	return output;
}