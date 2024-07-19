#include "NN_socket_interpreter.h"

NN_manager::NN_manager()
{

}

return_specifier NN_manager::parse_message(void* message, size_t message_length)
{
	return_specifier output = return_specifier {0, 0, 0};
	action_enum action = (action_enum)(*(size_t*)message);
	size_t offset = sizeof(size_t);
	switch (action)
	{
		case construct:
			size_t layer_count = *(size_t*)(message + offset);
			offset += sizeof(size_t);

			break;
	}
}
