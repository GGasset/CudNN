#include "NN_client.h"
#include <uchar.h>

NN_client::NN_client()
{
    socket = socket_client();
}

NN_client::~NN_client()
{
    if (network_id < 0) return;
    //TODO: ask for network destruction
}

void NN_client::link_NN(NN_constructor_client constructor)
{
    if (network_id >= 0) return;

   	size_t message_length = sizeof(size_t) * 3 + sizeof(bool) + constructor.layer_count * sizeof(size_t) * 4;
	void* message = new char[message_length];

	size_t offset = 0;
	*(size_t*)(message + offset) = action_enum::construct;
	offset += sizeof(size_t);

	*(size_t*)(message + offset) = constructor.layer_count;
	offset += sizeof(size_t);

	for (size_t i = 0; i < constructor.layer_count; i++)
	{
		*(size_t*)(message + offset) = constructor.connection_types[i];
		offset += sizeof(size_t);

		*(size_t*)(message + offset) = constructor.neuron_types[i];
		offset += sizeof(size_t);

		*(size_t*)(message + offset) = constructor.layer_lengths[i];
		offset += sizeof(size_t);

		*(size_t*)(message + offset) = constructor.activations[i];
		offset += sizeof(size_t);
	}
	*(size_t*)(message + offset) = constructor.input_length;
	offset += sizeof(size_t);

	*(bool*)(message + offset) = constructor.stateful;
	offset += sizeof(bool);
}
