
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
	srand(101);
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
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 30, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 15, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 7, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, 5, ActivationFunctions::sigmoid)
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_length, ActivationFunctions::sigmoid)
		.construct(input_length);
	/*
	NN* n = NN_constructor()
		.append_layer(ConnectionTypes::Dense, NeuronTypes::Neuron, output_length, ActivationFunctions::sigmoid)
		.construct(input_length);
	*/
	n->stateful = true;
	
	data_t total_mean_r = 0;
	data_t max_total_mean_r = -100000;
	size_t total_mean_r_count = 0;

	const size_t epochs = 100000;
	const size_t max_steps = 40;
	for (size_t i = 0; i < epochs; i++)
	{
		// RL demonstration
		data_t x = 0;
		data_t y = 0;

		data_t target_x = (rand() % 50 + 50);
		data_t target_y = (rand() % 50 + 50);
		target_x *= 1 - 2 * (rand() % 2);
		target_y *= 1 - 2 * (rand() % 2);

		data_t X[2];
		data_t *Y = 0;
		data_t* execution_values = 0;
		data_t* activations = 0;
		
		data_t rewards[max_steps];
		data_t supervised_outputs[max_steps * 2];
		
		bool success = false;
		size_t actual_steps = 0;
		data_t max_reward = 0;
		data_t mean_reward = 0;
		data_t mean_output[output_length] {0, 0};
		for (size_t step_i = 0; step_i < max_steps; step_i++)
		{
			
			target_x = (rand() % 50 + 50);
			target_y = (rand() % 50 + 50);
			target_x *= 1 - 2 * (rand() % 2);
			target_y *= 1 - 2 * (rand() % 2);

			actual_steps++;
			rewards[step_i] = 0;

			data_t target_direction_x = target_x - x;
			data_t target_direction_y = target_y - y;
			data_t target_distance = abs(target_direction_x) + abs(target_direction_y);


			X[0] = (target_direction_x > 0 ? .5 : -.5) * 1;
			X[1] = (target_direction_y > 0 ? .5 : -.5) * 1;

			supervised_outputs[step_i * 2] = target_direction_x > 0 ? .75 : .25;
			supervised_outputs[step_i * 2 + 1] = target_direction_y > 0 ? .75 : .25;
			
			n->training_execute(
				1, X, &Y, true,
				&execution_values, &activations,
				step_i
			);

			//x += (Y[0] - .5) * 2 * 3;
			//y += (Y[1] - .5) * 2 * 3;
			x += (Y[0] > .5) ? 1 : -1;
			y += (Y[1] > .5) ? 1 : -1;

			mean_output[0] += Y[0];
			mean_output[1] += Y[1];


			data_t new_target_direction_x = target_x - x;
			data_t new_target_direction_y = target_y - y;
			data_t new_target_distance = abs(new_target_direction_x) + abs(new_target_direction_y);

			//rewards[step_i]--;
			rewards[step_i] += 
				((abs(target_direction_x) > abs(new_target_direction_x))
				+ (abs(target_direction_y) > abs(new_target_direction_y)))
				* .5;
			
			rewards[step_i] -= rewards[step_i] == 0;

			max_reward += (abs(rewards[step_i]) - max_reward) * (abs(rewards[step_i]) > max_reward);
			mean_reward += rewards[step_i];

			if (success = abs(new_target_distance) < 3 && false)
			{
				printf("Hit!   ");
				break;
			}
		}
		
		mean_output[0] /= actual_steps;
		mean_output[1] /= actual_steps;
		
		mean_reward /= actual_steps;

		data_t value_functions[max_steps];
		const data_t gamma = .995;
		for (size_t j = 0; j < actual_steps; j++)
		{
			value_functions[j] = 0;
			data_t discount_factor = 1;
			
			for (size_t k = j; k < actual_steps; k++, discount_factor *= gamma)
			{
				value_functions[j] += rewards[k] * discount_factor;
			}
		}

		// Denoted as greek lowercase delta
		data_t deltas[max_steps];
		for (size_t j = 0; j < actual_steps; j++)
		{
			data_t discount_factor = gamma;
			deltas[j] = -value_functions[j] + rewards[j] + discount_factor * (j == actual_steps - 1 ? 0 : value_functions[j + 1]);
		}

		const data_t lambda = .98;
		data_t gamma_lambda = gamma * lambda;
		data_t advantages[max_steps];
		for (size_t j = 0; j < actual_steps; j++)
		{
			advantages[j] = 0;
			data_t GAE_discount = 1;
			for (size_t k = j; k < actual_steps && (/*Handle lambda=0 for performace*/1); k++, GAE_discount *= gamma_lambda)
			{
				advantages[j] += GAE_discount * deltas[k];
			}
		}
		

		/*n->train(actual_steps,
			execution_values, activations, 
			advantages, true, actual_steps,
			CostFunctions::log_likelyhood, .5, 100, 0.2
		);*/

		n->train(actual_steps,
			execution_values, activations, 
			supervised_outputs, true, actual_steps * output_length,
			CostFunctions::MSE, .001, 100, .2
		);



		//printf("Mean reward: %.2f | final distance: %.2f | inital distance: %.2f || ", mean_reward, (abs(target_x - x) + abs(target_y - y)), abs(target_x) + abs(target_y));
		int reward_pos = (int)((1 + mean_reward) * 10);
		for (size_t j = 0; j <= 20; j++) reward_pos == j ? printf("#") : printf("-");
		
		total_mean_r += mean_reward;
		total_mean_r_count++;

		printf("  Current mean reward: %.7f | Mean Reward: %.2f | target_x: %.0f, target_y: %.0f | %i | %.3f", mean_reward, total_mean_r / total_mean_r_count, target_x, target_y, i, max_total_mean_r);
		printf(" | Mean x output: %.2f | Mean y output %.2f\n", mean_output[0], mean_output[1]);

		if (i % 100 == 0)
		{
			max_total_mean_r += ((total_mean_r / total_mean_r_count) - max_total_mean_r) * (total_mean_r > max_total_mean_r);
			total_mean_r = total_mean_r_count = 0;
		}
		delete[] Y;
	}
	delete n;
	cudaDeviceReset();
	//n.deallocate();
}
