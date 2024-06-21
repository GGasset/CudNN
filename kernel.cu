
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "NN.h"
#include "NeuronLayer.h"
#include "LSTMLayer.h"
#include "DenseConnections.h"
#include "NeatConnections.h"

/*
static int increment_i(size_t size, int to_increment, size_t i)
{
	i += to_increment;
	i -= (i - i % size) * (i >= size);
	i = (i % size) * (i < 0) + i * (i >= 0);
	return i;
}*/

int main()
{
	cudaSetDevice(0);

	const size_t t_count = 1;
	const size_t input_length = 32;
	const size_t output_length = 34;
	data_t X[input_length * t_count]{};
	data_t Y_hat[output_length * t_count]{};

	for (size_t t = 0; t < t_count; t++)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			X[t * input_length + i] = .05;//(.3) / t_count * (t + 1) + .2 / t_count;
		}
		for (size_t i = 0; i < output_length; i++)
		{
			Y_hat[t * output_length + i] = .5;//(.2) / t_count * (t + 1) + .2 / t_count + (.2) / t_count * (i + 1) / output_length;
		}
	}

	const size_t shape_length = 4;
	size_t shape[shape_length]{ input_length, 38, 33, output_length };
	ILayer** layers = new ILayer * [shape_length - 1];

	size_t previous_layer_start = 0;
	for (size_t i = 1; i < shape_length; i++)
	{
		IConnections* connections = new DenseConnections(previous_layer_start, shape[i - 1], shape[i]);
		layers[i - 1] = new LSTMLayer(connections, shape[i]);
		//layers[i - 1] = new NeuronLayer(connections, shape[i], ActivationFunctions::sigmoid);
		previous_layer_start += shape[i - 1];
	}

	NN n = NN(true, layers, input_length, shape_length - 1);
	n.stateful = true;
	for (size_t i = 0; i < 5000; i++)
	{
		//printf("\n\n\n");

		data_t* y = 0;
		//y = n.execute(X);
		n.supervised_train(t_count, X, Y_hat, true, CostFunctions::MSE, .005 / t_count, &y, true, 200000, 0);
		for (size_t t = 0; t < t_count; t++)
		{
			for (size_t j = 0; j < output_length; j++)
			{
				printf("   (%i)%f|%f   ", j, y[t * output_length + j], Y_hat[t * output_length + j]);
			}
			printf("\n-----------------------\n");
		}
		std::cout << std::endl << std::endl;
		//if (i % 100 == 0)
		//	n.delete_memory();
		delete[] y;
	}
	//n.deallocate();
}
