#include "unix_sock_interop.h"

int main(int argc, char *argv[])
{
	int socket_fd = connect(argv[0]);
	if (!socket_fd)
		return 1;

}

int connect(char file_name[])
{
	printf("Setting up socket...\n");
	int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	
	sockaddr_un address;
	address.sun_family = AF_UNIX;
	address.sun_path = file_name;
	
	if (bind(socket_fd, (struc sockaddr*)&address, sizeof(address)))
	{
		printf("bind error\n");
		return 0;
	}
	printf("bind success\n");
	return fd;
}
