#include <sys/select.h>
#include <vector>

#include "unix_sock_interop.h"

int main(int argc, char *argv[])
{
	int socket_fd = connect();
	if (!socket_fd)
		return 1;

	NN_manager networks = NN_manager(1000);
	std::vector<int> clients;
	auto pending_messages_sizes = HashTable<int, size_t>(200);
	auto pending_out_messages = HashTable<int, return_specifier>(200);
	printf("Waiting for connections...\n");
	try
	{
		while (true)
		{
			fd_set read_fds;
			fd_set write_fds;

			FD_ZERO(&read_fds);
			FD_ZERO(&write_fds);
			
			FD_SET(socket_fd, &read_fds); // Add fd to set

			fd_set set_cpy = set;
			int max_fd = socket_fd;
			for (auto it = clients.begin(); it != clients.end(); it++)
			{
				size_t client_fd = *it;
				FD_SET(client_fd, &read_fds);
				max_fd = max_fd * (max_fd >= client_fd) + client_fd * (client_fd > max_fd);
			}

			SinglyLinkedListNode<size_t>* pending_messages_fds = pending_out_messages.GetKeys();
			for (auto it = pending_messages_fds; it; it = it->next)
				if (!FD_ISSET(it->value, &write_fds))
					FD_SET(it->value, &write_fds);
			pending_messages_fds->free();

			if (select(max_fd, &read_fds, 0, 0, 0) < 0)
			{
				printf("select error\n");
				throw;
			}

			if (FD_ISSET(socket_fd, &read_fds))
			{
				size_t new_client_fd = accept(socket_fd, 0, 0);
				if (new_client_fd < 0)
				{
					printf("accept_error\n");
					throw;
				}

				clients.push_back(new_client_fd);
				printf("Client connected\n");
			}

			for (auto fd_it = clients.begin(); fd_it != clients.end(); fd_it++)
			{
				size_t fd = *fd_it;
				if (FD_ISSET(fd, &read_fds))
				{
					bool has_sent_message_size = false;
					size_t bytes_to_read = pending_messages_sizes.Get(fd, has_sent_message_size);
					bytes_to_read += sizeof(size_t) * (!has_sent_message_size);
					
					void* message = malloc(bytes_to_read);
					read(fd, message, bytes_to_read);
					if (has_sent_message_size)
					{
						pending_messages_sizes.Remove(fd);
						return_specifier returned = networks.parse_message(message, bytes_to_read);
						pending_out_messages.Add(fd, returned);
					}
					else
						pending_messages_sizes.Add(*(size_t*)message);
					
				}
				if (FD_ISSET(fd, &write_fds))
				{
					return_specifier = pending_out_messages.Get(fd);
					write(fd, &return_specifier, sizeof(return_specifier));
					delete return_specifier.return_value;
					pending_out_messages.Remove(fd);
				}
			}
		}
	}
	finally
	{
		close(socket_fd);
		auto out_messages_keys = pending_out_messages.GetKeys();
		for (SinglyLinkedListNode<return_specifier> it = out_message_keys; it; it = it->next)
			delete[] it->value.return_value;
	}
}

int connect()
{
	printf("Setting up socket...\n");
	int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);

	sockaddr_un address;
	address.sun_family = AF_UNIX;

	address.sun_path[0] = '\0';
	const char bind_path[] = "NN_socket";
	strncpy(address.sun_path + 1, bind_path, sizeof(address.sun_path) - 1);
	
	if (bind(socket_fd, (struct sockaddr*)&address, sizeof(address)))
	{
		printf("bind error\n");
		return 0;
	}
	printf("binding to \"%s\" abstract name succesful\n", bind_path);
	if (listen(socket_fd, 1024))
	{
		printf("listen error\n");
		return 0;
	}
	return socket_fd;
}
