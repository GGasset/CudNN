#ifdef INCLUDE_BACKEND

#include <stdio.h>
#include "ILayer.h"
#include "costs.cuh"
#include "functionality.h"

#include "NeatConnections.h"
#include "NeuronLayer.h"
#include "LSTMLayer.h"
#include "kernel_macros.h"

#endif

#include "NN_enums.h"
#include "costs.cuh"
#include "neuron_operations.cuh"
#include "evolution_info.h"

#pragma once
class ILayer;

class NN
{
private:
	ILayer **layers = 0;
	size_t layer_count = 0;
	size_t neuron_count = 0;
	size_t input_length = 0;
	size_t output_length = 0;
	size_t* output_activations_start = 0;
	size_t execution_value_count = 0;
	size_t derivative_count = 0;
	short contains_recurrent_layers = 0;
	size_t gradient_count = 0;

	// Now state derivatives are 1 (variable) by default so no need to save derivatives
	//data_t* activations_since_memory_deletion = 0;
	//data_t* execution_values_since_memory_deletion = 0;
	//data_t* derivatives_since_memory_deletion = 0;
	//size_t since_memory_deletion_t_count = 0;


protected:
	void set_fields();

public:
	NN();

	evolution_metadata evolution_values;
	bool stateful = false;

	size_t get_input_length();
	size_t get_output_length();

	~NN();
	NN(ILayer** layers, size_t input_length, size_t layer_count);

	void execute(data_t* input, data_t* execution_values, data_t *activations, size_t t, data_t* output_start_pointer, short copy_output_to_host);
	void set_up_execution_arrays(data_t** execution_values, data_t** activations, size_t t_count);
	data_t* batch_execute(data_t* input, size_t t_count);
	data_t* inference_execute(data_t* input);

	data_t adjust_learning_rate(
		data_t learning_rate,
		data_t cost,
		LearningRateAdjusters adjuster,
		data_t max_learning_rate,
		data_t previous_cost = 0
	);

	data_t calculate_output_costs(
		CostFunctions cost_function,
		size_t t_count,
		data_t* Y_hat,
		data_t* activations, size_t activations_start,
		data_t* costs, size_t costs_start
	);

	void training_execute(
		size_t t_count,
		data_t* X,
		data_t** Y,
		bool copy_Y_to_host,
		data_t** execution_values,
		data_t** activations,
		size_t old_arrays_t_length = 0
	);

	data_t train(
		size_t t_count,
		data_t* execution_values,
		data_t* activations,
		data_t* Y_hat,
		bool is_Y_hat_on_host_memory,
		size_t Y_hat_value_count,
		CostFunctions cost_function,
		data_t learning_rate,
		data_t gradient_clip,
		float dropout_rate
	);

	data_t training_batch(
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
		float dropout_rate = .2
	);

	/// <param name="gradients">- pointer to cero and to a valid array are valid</param>
	void backpropagate(
		size_t t_count,
		data_t* costs,
		data_t* activations,
		data_t* execution_values,
		data_t** gradients
	);

	void calculate_derivatives(
		data_t* activations, size_t activations_start,
		data_t* derivatives, size_t previous_derivatives_start, size_t derivatives_start,
		data_t* execution_values, size_t execution_values_start
	);

	void calculate_gradients(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start,
		data_t* costs, size_t costs_start,
		data_t* gradients, size_t gradients_start, size_t next_gradients_start,
		data_t* derivatives, size_t derivatives_start, size_t previous_derivatives_start
	);

	void subtract_gradients(
		data_t* gradients, size_t gradients_start, data_t learning_rate, float dropout_rate, data_t gradient_clip
	);

	void evolve();
	void add_layer(size_t insert_i, ILayer* layer);
	void add_output_neuron();
	void add_input_neuron();
	void add_neuron(size_t layer_i);
	
	/// <param name="neuron_i">in respect to the whole network</param>
	void adjust_to_added_neuron(int layer_i, size_t neuron_i);
	void remove_neuron(size_t layer_i);
	void remove_neuron(size_t layer_i, size_t layer_neuron_i);


	void delete_memory();

	NN clone();

	void deallocate();

	void print_shape();
};
