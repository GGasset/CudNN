#include "client.h"

socket_client::socket_client()
{
	fd = 0;
	int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (socket_fd == -1)
		throw;

        const char bind_path[] = "NN_socket_client";
        const char server_bind_path[] = "NN_socket";

        sockaddr_un address;
        address.sun_family = AF_UNIX;
        address.sun_path[0] = '\0';
        strncpy(address.sun_path + 1, bind_path, sizeof(address.sun_path) - 1);	

	sockaddr_un server_address;
	server_address.sun_family = AF_UNIX;
	server_address.sun_path[0] = '\0';
	strncpy(server_address.sun_path + 1, server_bind_path, sizeof(server_address.sun_path) - 1);

	int bind_result = 0;
	//if (bind_result = bind(socket_fd, (struct sockaddr*)&address, sizeof(address))) throw;

	int connect_result = 0;
	if (connect_result = connect(socket_fd, (struct sockaddr*)&server_address, sizeof(server_address))) throw;

	fd = socket_fd;
}

socket_client::~socket_client()
{
	size_t disconnect_message[1] { 0 };
	write(fd, disconnect_message, sizeof(size_t));
	close(fd);
}

return_specifier socket_client::send_message(void* message, size_t message_length)
{
	write(fd, &message_length, sizeof(size_t));
	write(fd, message, message_length);
	
	return_specifier output;
	output.return_value = 0;
	output.value_count = 0;
	output.error = 0;

	void* raw_output = &output;
	size_t read_count = read(fd, raw_output + sizeof(data_t*), sizeof(return_specifier) - sizeof(data_t*));
	if (output.value_count)
	{
		output.return_value = new data_t[output.value_count];
		read(fd, output.return_value, sizeof(data_t) * output.value_count);
	}
	return output;
}
