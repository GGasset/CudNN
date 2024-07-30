
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "NN_constructor.h"

/*
static int increment_i(size_t size, int to_increment, size_t i)
{
	i += to_increment;+
	i -= (i - i % size) * (i >= size);
	i = (i % size) * (i < 0) + i * (i >= 0);
	return i;
}*/

template<typename t>
t abs(t a)
{
	return a * (-1 + 2 * (a > 0));
}

int main()
{
	cudaSetDevice(0);

	const size_t t_count = 5;
	const size_t input_length = 10;
	const size_t output_length = 3;
	data_t X[input_length * t_count]{};
	data_t Y_hat[output_length * t_count]{};

	for (size_t t = 0; t < t_count; t++)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			X[t * input_length + i] = (.2);//  / t_count * ((t + 1)) + .2 / t_count;
		}
		 for (size_t i = 0; i < output_length; i++)
		{
			Y_hat[t * output_length + i] = .05;//(.2) / t_count * (t + 1) + .2 / t_count + (.2) / t_count * (i + 1) / output_length;
		}
	}



	NN* n = NN_constructor()
		.append_layer(ConnectionTypes::NEAT, NeuronTypes::Neuron, 5, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::NEAT, NeuronTypes::LSTM, output_length)
		.construct(input_length);
	n.stateful = true;
	data_t *prev_y = 0;
	for (size_t i = 0; i < 1000; i++)
	{
		//printf("\n\n\n");
		data_t* y = 0;
		n->training_batch(t_count, X, Y_hat, true, output_length * t_count, CostFunctions::MSE, .05 / t_count, &y, true, 200000, 0);
		//n->print_shape();
		
		// Reinforcement learning
		/*data_t* reward = new data_t[t_count];
		for (size_t i = 0; i < t_count; i++)
			reward[i] = 0;
		data_t* execution_values = 0;
		data_t* activations = 0;
		n.training_execute(
			t_count,
			X,
			&y,
			true,
			&execution_values,
			&activations
		);
		for (size_t i = 0; i < output_length * t_count && prev_y; i++)
		{
			//reward[i / output_length] += (prev_y ? (abs(Y_hat[i] - prev_y[i]) > abs(Y_hat[i] - y[i]) ? .01 : -.01) : 0) / (output_length);
			reward[i / output_length] += 1 - abs(Y_hat[i] - y[i]);// - (1 - abs(Y_hat[i] - prev_y[i])) - (1 - abs(y[i] - prev_y[i]));
		}
		n.train(
			t_count,
			execution_values,
			activations,
			reward,
			true,
			t_count,
			CostFunctions::log_likelyhood,
			.001,
			2000,
			0
		);
		delete[] reward;*/
		
		for (size_t t = 0; t < t_count; t++)
		{
			for (size_t j = 0; j < output_length; j++)
			{
				printf("   (%i)%f|%f   ", j, y[t * output_length + j], Y_hat[t * output_length + j]);
			}
			printf("\n-----------------------\n");
		}
		std::cout << std::endl << std::endl;
		if (prev_y) delete[] prev_y;
		prev_y = y;
		//delete[] y;
		//n.evolve();
		//if (i % 100 == 0)
		//n.delete_memory();
	}
	delete[] prev_y;
	delete n;
	//n.deallocate();
}
