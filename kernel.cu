
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "NN.h"
#include "NeuronLayer.h"
#include "LSTMLayer.h"
#include "DenseConnections.h"
#include "NEATConnections.h"

static int increment_i(size_t size, int to_increment, size_t i)
{
	i += to_increment;
	i -= (i - i % size) * (i >= size);
	i = (i % size) * (i < 0) + i * (i >= 0);
	return i;
}

int main()
{
	cudaSetDevice(0);

	const size_t t_count = 2;
	const size_t input_length = 2;
	const size_t output_length = 3;
	data_t X[input_length * t_count]{};
	data_t Y_hat[output_length * t_count]{};

	for (size_t t = 0; t < t_count; t++)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			X[t * input_length + i] = (.3) / t_count * (t + 1) + .2 / t_count;
		}
		for (size_t i = 0; i < output_length; i++)
		{
			Y_hat[t * output_length + i] = (.2) / t_count * (t + 1) + .2 / t_count + (.2) / t_count * (i + 1) / output_length;
		}
	}

	const size_t shape_length = 3;
	size_t shape[shape_length]{ input_length, 3, output_length };
	ILayer** layers = new ILayer * [shape_length - 1];

	size_t previous_layer_start = 0;
	for (size_t i = 1; i < shape_length; i++)
	{
		IConnections* connections = new DenseConnections(previous_layer_start, shape[i - 1], shape[i]);
		layers[i - 1] = new LSTMLayer(connections, shape[i]);
		previous_layer_start += shape[i - 1];
	}

	NN n = NN(true, layers, input_length, shape_length - 1, 0);
	n.stateful = true;
	for (size_t i = 0; i < 5000; i++)
	{
		//printf("\n\n\n");

		data_t* y = 0;//n.execute(X);
		n.supervised_train(t_count, X, Y_hat, true, CostFunctions::MSE, .0001 / t_count, &y, true, 200000, .2);
		for (size_t t = 0; t < t_count; t++)
		{
			for (size_t j = 0; j < output_length; j++)
			{
				printf("%f | %f  ", y[t * output_length + j], Y_hat[t * output_length + j]);
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
