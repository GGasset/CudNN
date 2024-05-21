
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "NN.h"
#include "DenseNeuronLayer.h"

int main()
{
	const size_t input_length = 1;
	const size_t output_length = 1;
	data_t X[input_length];
	data_t Y_hat[output_length];

	const size_t shape_length = 3;
	size_t shape[shape_length]{ input_length, 3, output_length };
	ILayer** layers = new ILayer * [shape_length - 1];

	size_t gradient_count = 0;
	size_t neuron_count = 0;
	for (size_t i = 1; i < shape_length; i++)
	{
		layers[i - 1] = new DenseNeuronLayer(gradient_count, shape[i], neuron_count, shape[i - 1], ActivationFunctions::sigmoid);
		gradient_count += layers[i - 1]->layer_gradient_count;
		neuron_count += shape[i - 1];
	}

	NN n = NN(false, layers, input_length, shape_length - 1, 0);
	for (size_t i = 0; i < 100; i++)
	{
		for (size_t j = 0; j < input_length; j++)
		{
			X[j] = 1;
			//printf("%f ", X[j]);
		}
		for (size_t j = 0; j < output_length; j++)
		{
			Y_hat[j] = .5;
		}
		//printf("\n\n\n");

		data_t* y = 0;//n.execute(X);
		n.supervised_train(1, X, Y_hat, true, CostFunctions::MSE, &y, true);
		for (size_t j = 0; j < output_length; j++)
		{
			printf("%f  ", y[j]);
		}
		std::cout << std::endl;
		delete[] y;
	}
	//n.deallocate();
}