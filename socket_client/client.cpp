#include "client.h"

socket_client::socket_client()
{
	fd = 0;
	int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (socket_fd == -1)
		throw;

	sockaddr_un server_address;
	server_address.sun_family = AF_UNIX;
	//server_address.sun_path[0] = 0;
	strncpy(server_address.sun_path, BIND_PATH, sizeof(server_address.sun_path));

	int connect_result = 0;
	if (connect_result = connect(socket_fd, (struct sockaddr*)&server_address, sizeof(server_address)))
	{
		int a = WSAGetLastError();
		__close_sock(fd);
		CLEANUP
		throw;
	}

	fd = socket_fd;
}

socket_client::~socket_client()
{
	size_t disconnect_message[1] { 0 };
	send(fd, (char *)disconnect_message, sizeof(size_t), 0);
	__close_sock(fd);
}

return_specifier socket_client::send_message(void* message, size_t message_length)
{
	if (!message) throw;
	send(fd, (char *)&message_length, sizeof(size_t), 0);
	send(fd, (char *)message, message_length, 0);
	
	return_specifier output;
	output.return_value = 0;
	output.value_count = 0;
	output.error = 0;

	void* raw_output = &output;
	size_t read_count = recv(fd, (char *)raw_output + sizeof(data_t*), sizeof(return_specifier) - sizeof(data_t*), 0);
	if (output.value_count)
	{
		output.return_value = new data_t[output.value_count];
		recv(fd, (char *)output.return_value, sizeof(data_t) * output.value_count, 0);
	}
	return output;
}
