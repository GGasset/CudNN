
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

	const size_t input_length = 2;
	const size_t output_length = 2;
	/*const size_t t_count = 5;
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
	}*/



	NN* n = NN_constructor()
		//.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 3, ActivationFunctions::sigmoid)
		//.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 4, ActivationFunctions::sigmoid)
		//.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 3, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_length, ActivationFunctions::sigmoid)
		.construct(input_length);
	n->stateful = true;

	const size_t epochs = 10000;
	data_t epochs_mean_reward[epochs];
	const size_t max_steps = 70;
	for (size_t i = 0; i < epochs; i++)
	{
		// RL demonstration
		data_t x = 0;
		data_t y = 0;

		data_t target_x = rand() % 50 + 50;
		data_t target_y = rand() % 50 + 50;
		target_x *= 1 - 2 * (rand() % 2);
		target_y *= 1 - 2 * (rand() % 2);

		data_t X[2];
		data_t *Y = 0;
		data_t* execution_values = 0;
		data_t* activations = 0;
		
		data_t rewards[max_steps];
		
		bool success = false;
		size_t actual_steps = 0;
		data_t max_reward = 0;
		data_t mean_reward = 0;
		data_t mean_output[output_length] {0, 0};
		for (size_t step_i = 0; step_i < max_steps; step_i++)
		{
			actual_steps++;
			rewards[step_i] = 0;

			data_t target_direction_x = target_x - x;
			data_t target_direction_y = target_y - y;
			data_t target_distance = abs(target_direction_x) + abs(target_direction_y);

			X[0] = target_direction_x / 200 + .25;
			X[1] = target_direction_y / 200 + .25;

			n->training_execute(
				1, X, &Y, true,
				&execution_values, &activations,
				step_i
			);

			x += (Y[0] - .5) * 2 * 3;
			y += (Y[1] - .5) * 2 * 3;

			mean_output[0] += Y[0];
			mean_output[1] += Y[1];


			data_t new_target_direction_x = target_x - x;
			data_t new_target_direction_y = target_y - y;
			data_t new_target_distance = abs(new_target_direction_x) + abs(new_target_direction_y);

			rewards[step_i] += ((target_distance > new_target_distance) - (target_distance <= new_target_distance))/* * (1 / (1 + (new_target_distance / (target_x + target_y)))*/;
			//rewards[step_i] /= Y[0] + Y[1];

			max_reward += (abs(rewards[step_i]) - max_reward) * (abs(rewards[step_i]) > max_reward);
			mean_reward += rewards[step_i];

			if (success = abs(new_target_distance) < 3) break;
		}

		mean_reward /= actual_steps;
		epochs_mean_reward[i] = mean_reward;
		n->train(actual_steps,
			execution_values, activations, 
			rewards, true, actual_steps,
			CostFunctions::log_likelyhood, .03, 100, 0.05
		);
		//printf("Mean reward: %.2f | final distance: %.2f | inital distance: %.2f || ", mean_reward, (abs(target_x - x) + abs(target_y - y)), abs(target_x) + abs(target_y));
		int reward_pos = (int)(mean_reward * 10 + 10);
		printf("       ");
		for (size_t j = 0; j < 21; j++) reward_pos == j ? printf("#") : printf("-");
		
		data_t total_mean_r = 0;
		for (size_t j = 0; j < i + 1; j++) total_mean_r += epochs_mean_reward[j];

		printf("  Total mean reward: %.5f | %i\n", total_mean_r / (i + 1), i);

		delete[] Y;
	}
	delete n;
	//n.deallocate();
}
