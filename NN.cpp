﻿#ifndef NN_DEFINITIONS
#define NN_DEFINITIONS

#include "NN.h"

NN::NN(short contains_recurrent_layers, ILayer** layers, size_t input_length, size_t layer_count, size_t max_layer_count)
{
	// set max layer count to layer count if max_layer_count is lesser than layer count
	max_layer_count += (layer_count - max_layer_count) * (max_layer_count < layer_count);
	
	this->contains_recurrent_layers = contains_recurrent_layers;
	this->layers = layers;
	this->input_length = input_length;
	this->layer_count = layer_count;
	this->max_layer_count = layer_count;
	set_fields();
}

NN::~NN()
{
	deallocate();
}

void NN::set_fields()
{
	output_length = layers[layer_count - 1]->neuron_count;

	size_t neuron_count = input_length;
	size_t execution_value_count = 0;
	size_t derivative_count = 0;
	size_t gradient_count = 0;
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* layer = layers[i];

		layer->layer_activations_start = neuron_count;
		neuron_count += layer->neuron_count;

		layer->execution_values_layer_start = execution_value_count;
		execution_value_count += layer->execution_values_per_neuron * layer->neuron_count;

		layer->layer_derivatives_start = derivative_count;
		derivative_count += layer->layer_derivatives_start;

		layer->layer_gradients_start = gradient_count;
		gradient_count += layer->layer_gradient_count;
	}
	this->neuron_count = neuron_count;
	output_activations_start = &(layers[layer_count - 1]->layer_activations_start);
	this->execution_value_count = execution_value_count;
	this->derivative_count = derivative_count;
	this->gradient_count = gradient_count;
}

void NN::execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, short copy_output_to_host = true)
{
	cudaMemcpy(activations + t * neuron_count, input + input_length * t, sizeof(data_t) * input_length, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->execute(activations, neuron_count * t, execution_values, execution_value_count * t);
		cudaDeviceSynchronize();
	}
	if (copy_output_to_host)
	{
		cudaMemcpy(output_start_pointer + output_length * t, activations + neuron_count * t + *output_activations_start, sizeof(data_t) * output_length, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
}

void NN::set_up_execution_arrays(data_t** execution_values, data_t** activations, size_t t_count)
{
	cudaMalloc(&execution_values, sizeof(data_t) * execution_value_count * t_count);
	cudaMalloc(&activations, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
	cudaMemset(execution_values, 0, sizeof(data_t) * execution_value_count * t_count);
	cudaMemset(activations, 0, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
}

data_t* NN::execute(data_t* input, size_t t_count)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	set_up_execution_arrays(&execution_values, &activations, t_count);

	data_t* outputs = new data_t[output_length * t_count];
	for (size_t i = 0; i < output_length * t_count; i++)
	{
		outputs[i] = 0;
	}
	for (size_t i = 0; i < t_count; i++)
	{
		execute(input, execution_values, activations, i, outputs + output_length * i, 1);
	}


	cudaFree(execution_values);
	cudaFree(activations);
	cudaDeviceSynchronize();
	return outputs;
}

data_t* NN::execute(data_t* input)
{
	return execute(input, 1);
}

void NN::calculate_supervised_output_costs_gradients(
	CostFunctions cost_function,
	size_t t_count,
	data_t* Y_hat,
	data_t* activations, size_t activations_start,
	data_t* costs, size_t costs_start
)
{
	switch (cost_function)
	{
	case MSE:
		MSE_derivative kernel(t_count, output_length) (
			activations, neuron_count, activations_start, *output_activations_start,
			costs, costs_start,
			Y_hat, output_length
		);
		break;
	default:
		break;
	}
}

double NN::supervised_train(
	size_t t_count,
	data_t* X,
	data_t* Y_hat,
	CostFunctions cost_function,
	data_t** Y,
	bool copy_Y_to_host
)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	set_up_execution_arrays(&execution_values, &activations, t_count);

	data_t* costs = 0;
	cudaMalloc(&costs, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();

	cudaMemset(costs, 0, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();

	if (copy_Y_to_host)
	{
		cudaMalloc(Y, sizeof(data_t) * output_length * t_count);
		cudaDeviceSynchronize();
		cudaMemset(*Y, 0, sizeof(data_t) * output_length * t_count);
		cudaDeviceSynchronize();
	}

	for (size_t t = 0; t < t_count; t++)
	{
		execute(X, execution_values, activations, t, *Y, copy_Y_to_host);
	}
	calculate_supervised_output_costs_gradients(cost_function, t_count, Y_hat, activations, 0, costs, 0);
	cudaDeviceSynchronize();

	data_t* gradients = 0;
	backpropagate(
		t_count, costs, activations, execution_values, &gradients
	);

	for (size_t t = 0; t < t_count; t++)
	{
		subtract_gradients(gradients, gradient_count * t);
	}
}

void NN::backpropagate(
	size_t t_count, 
	data_t* costs,
	data_t* activations, 
	data_t* execution_values,
	data_t** gradients
)
{
	data_t* derivatives = 0;
	if (!*gradients)
		cudaMalloc(gradients, sizeof(data_t) * t_count * gradient_count);
	if (derivative_count)
		cudaMalloc(&derivatives, sizeof(data_t) * t_count * derivative_count);

	size_t activations_start = 0;
	size_t execution_values_start = 0;
	size_t derivatives_start = 0;
	size_t gradients_start = 0;
	for (size_t t = 0; t < t_count; t++)
	{
		activations_start = neuron_count * t;
		derivatives_start = derivative_count * t;
		execution_values_start = execution_value_count * t;
		calculate_derivatives(
			activations, activations_start, 
			derivatives, derivatives_start - derivative_count, derivatives_start,
			execution_values, execution_values_start
		);
	}
	for (int t = t_count - 1; t >= 0; t--)
	{
		gradients_start = gradient_count * t;
		size_t next_gradient_start = gradients_start + gradient_count;
		next_gradient_start -= next_gradient_start * (t == t_count - 1);

		derivatives_start = derivative_count * t;
		activations_start = neuron_count * t;

		calculate_gradients(
			activations, activations_start,
			execution_values, execution_values_start,
			costs, activations_start,
			*gradients, gradients_start, next_gradient_start,
			derivatives, derivatives_start, derivatives_start - derivative_count
		);
	}

	if (!stateful && contains_recurrent_layers)
		delete_memory();
}

void NN::calculate_derivatives(
	data_t* activations, size_t activations_start,
	data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
	data_t* execution_values, size_t execution_values_start
)
{
	// Todo: make layer gradient calculation async
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->calculate_derivatives(
			activations, activations_start,
			derivatives, previous_derivatives_start, derivatives_start,
			execution_values, execution_values_start
		);
		cudaDeviceSynchronize();
	}
}

void NN::calculate_gradients(
	data_t* activations, size_t activations_start,
	data_t* execution_values, size_t execution_values_start,
	data_t* costs, size_t costs_start, 
	data_t* gradients, size_t gradients_start, size_t next_gradients_start, 
	data_t* derivatives, size_t derivatives_start, size_t previous_derivatives_start
)
{
	for (int i = layer_count - 1; i >= 0; i--)
	{
		layers[i]->calculate_gradients(
			activations, activations_start,
			execution_values, execution_values_start,
			derivatives, derivatives_start,
			gradients, next_gradients_start, gradients_start,
			costs, costs_start
		);
		cudaDeviceSynchronize();
	}
}

void NN::subtract_gradients(data_t* gradients, size_t gradients_start)
{
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->subtract_gradients(gradients, gradients_start);
	}
	cudaDeviceSynchronize();
}

void NN::delete_memory()
{
	for (size_t i = 0; i < layer_count; i++)
		layers[i]->delete_memory();
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