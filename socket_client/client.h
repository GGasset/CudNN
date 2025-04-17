
#include <cstdio>
#include <sys/types.h>

#ifdef _WIN32
	#include <WinSock2.h>
	#include <afunix.h>

	#define BIND_PATH "abstract_local_NN"
	#define __close_sock(__fd) closesocket(__fd)

#else
	#include <unistd.h>
	#include <sys/socket.h>
	#include <sys/un.h>

	#define BIND_PATH "tmp/NN_socket"
	#define __close_sock(__fd) close(__fd)

#endif

#include "NN_enums.h"

#pragma once
class socket_client
{
private:
	int fd;
public:
	socket_client();
	~socket_client();

	return_specifier send_message(void* message, size_t message_length);
};
