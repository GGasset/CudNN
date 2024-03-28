#include "ILayer.h"

#pragma once
class NN
{
private:
	ILayer **layers = 0;
	size_t max_layer_count = 0;
	size_t layer_count = 0;
	size_t neuron_count = 0;
	size_t input_length = 0;
	size_t output_length = 0;
	size_t* output_activations_start = 0;
	size_t execution_value_count = 0;
	size_t gradient_value_count = 0;

public:
	data_t* Execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, short copy_output_to_host = true);
	data_t* Execute(data_t* input, size_t t_count);
	data_t* Execute(data_t* input);
};

