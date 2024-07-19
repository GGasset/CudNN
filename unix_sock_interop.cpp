#include "unix_sock_interop.h"

int main(int argc, char *argv[])
{
	int socket_fd = connect();
	if (!socket_fd)
		return 1;

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
	return socket_fd;
}
