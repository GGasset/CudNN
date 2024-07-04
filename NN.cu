﻿#ifndef NN_DEFINITIONS
#define NN_DEFINITIONS

#include "NN.h"

NN::NN(ILayer** layers, size_t input_length, size_t layer_count)
{
	this->layers = layers;
	this->input_length = input_length;
	this->layer_count = layer_count;
	set_fields();
}

NN::NN()
{
}

NN::~NN()
{
	deallocate();
}

void NN::set_fields()
{
	output_length = layers[layer_count - 1]->get_neuron_count();

	size_t neuron_count = input_length;
	size_t execution_value_count = 0;
	size_t derivative_count = 0;
	size_t gradient_count = 0;
	contains_recurrent_layers = true;
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* layer = layers[i];
		
		contains_recurrent_layers = contains_recurrent_layers && layer->is_recurrent;

		layer->layer_activations_start = neuron_count;
		neuron_count += layer->get_neuron_count();

		layer->execution_values_layer_start = execution_value_count;
		execution_value_count += layer->execution_values_per_neuron * layer->get_neuron_count();

		layer->layer_derivatives_start = derivative_count;
		derivative_count += layer->layer_derivative_count;

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
	cudaMalloc(execution_values, sizeof(data_t) * execution_value_count * t_count);
	cudaMalloc(activations, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();
	cudaMemset(*execution_values, 0, sizeof(data_t) * execution_value_count * t_count);
	cudaMemset(*activations, 0, sizeof(data_t) * neuron_count * t_count);
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
		execute(input, execution_values, activations, i, outputs, 1);
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



data_t NN::calculate_output_costs(
	CostFunctions cost_function,
	size_t t_count,
	data_t* Y_hat,
	data_t* activations, size_t activations_start,
	data_t* costs, size_t costs_start
)
{
	data_t* cost = 0;
	cudaMalloc(&cost, sizeof(data_t));
	cudaDeviceSynchronize();
	cudaMemset(cost, 0, sizeof(data_t));
	cudaDeviceSynchronize();
	switch (cost_function)
	{
	case MSE:
		MSE_derivative kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, neuron_count, activations_start, *output_activations_start,
			costs, costs_start,
			Y_hat, output_length
		);
		MSE_cost kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, neuron_count, activations_start, *output_activations_start,
			Y_hat, output_length,
			cost
		);
		break;
	case log_likelyhood:
		log_likelyhood_derivative kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, activations_start,
			neuron_count, *output_activations_start, output_length,
			costs, costs_start,
			Y_hat
		);
		log_likelyhood_cost kernel(dim3(output_length / 32 + (output_length % 32 > 0), t_count), 32) (
			activations, neuron_count, activations_start, *output_activations_start,
			Y_hat, output_length,
			cost
		);
		break;
	default:
		break;
	}
	cudaDeviceSynchronize();
	multiply_array kernel(1, 1) (
		cost, 1, 1 / (output_length * t_count)
	);
	data_t host_cost = 0;
	cudaDeviceSynchronize();
	cudaMemcpy(&host_cost, cost, sizeof(data_t), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(cost);
	return host_cost;
}

void NN::training_execute(
	size_t t_count,
	data_t* X,
	data_t** Y,
	bool copy_Y_to_host,
	data_t** execution_values,
	data_t** activations,
	data_t** costs
)
{
	set_up_execution_arrays(execution_values, activations, t_count);

	cudaMalloc(costs, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();

	cudaMemset(*costs, 0, sizeof(data_t) * neuron_count * t_count);
	cudaDeviceSynchronize();

	if (copy_Y_to_host)
	{
		*Y = new data_t[output_length * t_count];
		for (size_t i = 0; i < output_length * t_count; i++)
		{
			(*Y)[i] = 0;
		}
	}

	for (size_t t = 0; t < t_count; t++)
	{
		execute(X, *execution_values, *activations, t, copy_Y_to_host ? *Y : 0, copy_Y_to_host);
	}
}


data_t NN::train(
	size_t t_count,
	data_t* execution_values
	data_t* activations,
	data_t* costs,
	data_t* Y_hat,
	bool is_Y_hat_on_host_memory,
	size_t Y_hat_value_count,
	CostFunctions cost_function,
	data_t learning_rate,
	data_t gradient_clip,
	float dropout_rate
)
{
	if (is_Y_hat_on_host_memory)
	{
		data_t* temp_Y_hat = 0;
		cudaMalloc(&temp_Y_hat, sizeof(data_t) * Y_hat_value_count);
		cudaMemcpy(temp_Y_hat, Y_hat, sizeof(data_t) * Y_hat_value_count, cudaMemcpyHostToDevice);
		Y_hat = temp_Y_hat;
	}
	data_t cost = calculate_output_costs(cost_function, t_count, Y_hat, activations, 0, costs, 0);
	cudaDeviceSynchronize();

	data_t* gradients = 0;
	backpropagate(
		t_count, costs, activations, execution_values, &gradients
	);

	for (size_t t = 0; t < t_count; t++)
	{
		subtract_gradients(gradients, gradient_count * t, learning_rate, dropout_rate, gradient_clip);
	}

	if (is_Y_hat_on_host_memory)
		cudaFree(Y_hat);
	cudaFree(activations);
	cudaFree(execution_values);
	cudaFree(costs);
	cudaFree(gradients);
	cudaDeviceSynchronize();

	return cost;
}

data_t NN::train(
	size_t t_count,
	data_t* X,
	data_t* Y_hat,
	bool is_Y_hat_on_host_memory,
	size_t Y_hat_value_count,
	CostFunctions cost_function,
	data_t learning_rate,
	data_t** Y,
	bool copy_Y_to_host,
	data_t gradient_clip,
	float dropout_rate
)
{
	data_t* execution_values = 0;
	data_t* activations = 0;
	data_t* costs = 0;
	training_execute(
		t_count,
		X,
		Y,
		copy_Y_to_host,
		&execution_values,
		&activations,
		&costs
	);
	return train(
		t_count, 
		execution_values,
		activations,
		costs,
		Y_hat,
		is_Y_hat_on_host_memory,
		Y_hat_value_count,
		cost_function,
		learning_rate,
		gradient_clip,
		dropout_rate
	);
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

void NN::subtract_gradients(data_t* gradients, size_t gradients_start, data_t learning_rate, float dropout_rate, data_t gradient_clip)
{
	for (size_t i = 0; i < layer_count; i++)
	{
		ILayer* current_layer = layers[i];
		size_t layer_length = current_layer->get_neuron_count();

		short* dropout = 0;
		float* normalized_random_samples = 0;
		cudaMalloc(&dropout, sizeof(short) * layer_length);
		cudaMalloc(&normalized_random_samples, sizeof(float) * layer_length);
		cudaDeviceSynchronize();
		
		cudaMemset(dropout, 0, sizeof(short) * layer_length);
		IConnections::generate_random_values(&normalized_random_samples, layer_length);
		cudaDeviceSynchronize();
		cud_set_dropout kernel(layer_length / 32 +  (layer_length % 32 > 0), 32) (dropout_rate, normalized_random_samples, dropout, layer_length);
		cudaDeviceSynchronize();

		current_layer->subtract_gradients(gradients, gradients_start, learning_rate, dropout, gradient_clip);

		cudaFree(dropout);
		cudaFree(normalized_random_samples);
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
}

void NN::evolve()
{
	for (size_t i = 0; i < layer_count; i++)
	{
		layers[i]->mutate_fields(evolution_values);
		layers[i]->connections->mutate_fields(evolution_values);
	}
	if (evolution_values.layer_addition_probability > get_random_float())
	{
		printf("Adding layer\n");
		NeuronTypes insert_type = (NeuronTypes)(rand() % NeuronTypes::last_neuron_entry);
		size_t insert_i = rand() % (layer_count - 1);
		
		size_t previous_layer_length = input_length;
		size_t previous_layer_activations_start = 0;
		if (insert_i)
		{
			ILayer* previous_layer = layers[insert_i];
			previous_layer_length = previous_layer->get_neuron_count();
			previous_layer_activations_start = previous_layer->layer_activations_start;
		}
		
		IConnections* new_connections = new NeatConnections(previous_layer_activations_start, previous_layer_length, 1);
		ILayer* new_layer = 0;

		switch (insert_type)
		{
		case NN::Neuron:
			new_layer = new NeuronLayer(new_connections, 1, (ActivationFunctions)(rand() % ActivationFunctions::activations_last_entry));
			break;
		case NN::LSTM:
			new_layer = new LSTMLayer(new_connections, 1);
			break;
		default:
			throw "Neuron_type not added to evolve method";
			break;
		}
		add_layer(insert_i, new_layer);
	}
	if (evolution_values.neuron_deletion_probability > get_random_float())
	{
		printf("removing neuron\n");
		size_t layer_i = rand() % (layer_count - 1);
		remove_neuron(layer_i);
	}
	if (evolution_values.neuron_addition_probability > get_random_float())
	{
		printf("adding_neuron\n");
		size_t layer_i = rand() % (layer_count - 1);
		add_neuron(layer_i);
	}
	float* evolution_values_pointer = (float*)(&evolution_values);
	for (size_t i = 0; i < sizeof(evolution_metadata) / sizeof(float); i++)
	{
		evolution_values_pointer[i] +=
			evolution_values.evolution_metadata_field_max_mutation *
			(evolution_values.evolution_metadata_field_mutation_chance > get_random_float()) *
			(1 - 2 * (get_random_float() > .5));
	}
}

void NN::add_layer(size_t insert_i, ILayer* layer)
{
	ILayer** tmp_layers = layers;
	layer_count++;

	// insert layer
	layers = new ILayer * [layer_count];
	for (size_t i = 0; i < insert_i; i++)
		layers[i] = tmp_layers[i];
	layers[insert_i] = layer;
	for (size_t i = insert_i + 1; i < layer_count; i++)
		layers[i] = tmp_layers[i - 1];

	// Update info
	set_fields();
	size_t added_neuron_count = layer->get_neuron_count();
	size_t added_layer_activations_start = layer->layer_activations_start;
	for (size_t i = 0; i < added_neuron_count; i++)
	{
		adjust_to_added_neuron(insert_i, added_layer_activations_start + i);
	}
	set_fields();
}

void NN::add_output_neuron()
{
	add_neuron(layer_count - 1);
}

void NN::add_input_neuron()
{
	for (size_t i = 0; i < layer_count; i++)
	{
		adjust_to_added_neuron(-1, input_length);
	}
	input_length++;
	set_fields();
}

void NN::add_neuron(size_t layer_i)
{

	size_t previous_layer_length = input_length;
	size_t previous_layer_activations_start = 0;
	if (layer_i)
	{
		ILayer *previous_layer = layers[layer_i];
		previous_layer_length = previous_layer->get_neuron_count();
		previous_layer_activations_start = previous_layer->layer_activations_start;
	}
	size_t added_neuron_i = layers[layer_i]->layer_activations_start + layers[layer_i]->get_neuron_count();
	layers[layer_i]->add_neuron(previous_layer_length, previous_layer_activations_start, 1, 0);
	adjust_to_added_neuron(layer_i, added_neuron_i);
	set_fields();
}

void NN::adjust_to_added_neuron(int layer_i, size_t neuron_i)
{
	size_t layer_distance_from_added_neuron = 1;
	for (int i = layer_i + 1; i < layer_count; i++, layer_distance_from_added_neuron++)
	{
		float connection_probability = 1.0 / layer_distance_from_added_neuron;
		connection_probability += (1 - connection_probability) * evolution_values.layer_distance_from_added_neuron_connection_addition_modifier;
		layers[i]->adjust_to_added_neuron(neuron_i, connection_probability);
	}
}

void NN::remove_neuron(size_t layer_i)
{
	if (layers[layer_i]->get_neuron_count() == 1)
		return;
	size_t layer_neuron_count = layers[layer_i]->get_neuron_count();
	remove_neuron(layer_i, rand() % layer_neuron_count);
}

void NN::remove_neuron(size_t layer_i, size_t layer_neuron_i)
{
	size_t removed_neuron_i = layers[layer_i]->layer_activations_start + layer_neuron_i;
	layers[layer_i]->remove_neuron(layer_neuron_i);
	for (size_t i = layer_i + 1; i < layer_count; i++)
		layers[i]->adjust_to_removed_neuron(removed_neuron_i);

	set_fields();
}

void NN::delete_memory()
{
	for (size_t i = 0; i < layer_count; i++)
		layers[i]->delete_memory();
}

NN NN::clone()
{
	NN clone;
	clone.layer_count = layer_count;
	clone.neuron_count = neuron_count;
	clone.input_length = input_length;
	clone.output_length = output_length;
	
	clone.layers = new ILayer*[layer_count];
	for (size_t i = 0; i < layer_count; i++)
	{
		clone.layers[i] = layers[i]->layer_specific_clone();
		layers[i]->ILayerClone(clone.layers[i]);
	}
	clone.set_fields();
	clone.evolution_values = evolution_values;
	clone.contains_recurrent_layers = contains_recurrent_layers;
	return clone;
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

void NN::print_shape()
{
	printf("%i ", input_length);
	for (size_t i = 0; i < layer_count; i++)
		printf("%i ", layers[i]->get_neuron_count());
	printf("\n");
}
#endif
