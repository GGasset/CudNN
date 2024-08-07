#include "NN_socket_interpreter.h"

NN_manager::NN_manager(size_t bucket_count)
{
	networks = new HashTable<size_t, network_container*>(bucket_count);
}

return_specifier* NN_manager::parse_message(void* message, size_t message_length)
{
	return_specifier* output = (return_specifier*)malloc(sizeof(return_specifier));
	output->return_value = 0;
	output->value_count = 0;
	output->error = 0;

	action_enum action = (action_enum)(*(size_t*)message);
	size_t offset = sizeof(size_t);
	switch (action)
	{
		case construct:
			{
				NN_constructor constructor = NN_constructor();
				size_t layer_count = *(size_t*)(message + offset);
				offset += sizeof(size_t);

				if (message_length != sizeof(size_t) * 3 + sizeof(bool) + layer_count * sizeof(size_t) * 4) throw;

				for (size_t i = 0; i < layer_count; i++)
				{
					ConnectionTypes connections = *(ConnectionTypes*)(message + offset);
					offset += sizeof(size_t);

					NeuronTypes neurons = *(NeuronTypes*)(message + offset);
					offset += sizeof(size_t);

					size_t neuron_count = *(size_t*)(message + offset);
					offset += sizeof(size_t);

					ActivationFunctions activation = *(ActivationFunctions*)(message + offset);
					offset += sizeof(size_t);

					constructor.append_layer(connections, neurons, neuron_count, activation);
				}
				size_t input_length = *(size_t*)(message + offset);
				offset += sizeof(size_t);

				bool stateful = *(bool*)(message + offset);
				offset += sizeof(bool);

				auto ids = networks->GetKeys();
				size_t network_id = 0;
				if (ids)
				{
				       	network_id = ids->max() + 1;
					ids->free();
				}

				network_container* network = (network_container*)malloc(sizeof(network_container));
				network->accumulated_training_t_count = 0;
				network->accumulated_activations = 0;
				network->accumulated_execution_values = 0;
				network->accumulated_Y_hat = 0;

				network->network = constructor.construct(input_length, stateful);

				networks->Add(network_id, network);
				network_count++;

				output->return_value = new data_t[1];
				output->return_value[0] = network_id;
				output->value_count = 1;
#ifdef DEBUG
				printf("Network created\n");
#endif
			}
			break;
		case destruct:
			{
				size_t id = *(size_t*)(message + offset);
				offset += sizeof(size_t);

				if (message_length != sizeof(size_t) * 2) throw;

				bool is_registered = false;
				network_container* network = networks->Get(id, is_registered);
				if (is_registered)
				{
					if (network->accumulated_activations) cudaFree(network->accumulated_activations);
					if (network->accumulated_execution_values) cudaFree(network->accumulated_execution_values);
					if (network->accumulated_Y_hat) cudaFree(network->accumulated_Y_hat);
					delete network->network;
					free(network);
#ifdef DEBUG
					printf("network destructed\n");
#endif
				}
				networks->Remove(id);
			}
			break;
		default:
			break;
	}
	if (offset > message_length) throw;
	return output;
}
