#include "NN_client.h"
#include <uchar.h>

NN_client::NN_client()
{
}

NN_client::~NN_client()
{
	if (network_id < 0) return;

	disconnect_NN();
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

	return_specifier response = socket.send_message(message, message_length);
	
	network_id = response.return_value[0];
	delete[] response.return_value;
	delete[] message;
}

void NN_client::disconnect_NN()
{
	if (network_id < 0) return;

	size_t message_length = sizeof(size_t) * 2;
	size_t* message = new size_t[2];
	message[0] = action_enum::destruct;
	message[1] = (size_t)network_id;
	socket.send_message(message, message_length);
}
