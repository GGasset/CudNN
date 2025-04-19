
#include <cstdio>
#include <sys/types.h>

#ifdef _WIN32
	#include <WinSock2.h>
	#include <afunix.h>
	
	#ifndef BIND_PATH
		#define BIND_PATH "C:\\Users\\Public\\socket.sock"
	#endif
	#define __close_sock(__fd) closesocket(__fd)
	#define CLEANUP WSACleanup();

#else
	#include <unistd.h>
	#include <sys/socket.h>
	#include <sys/un.h>

	#ifndef BIND_PATH
		#define BIND_PATH "/tmp/NN_socket"
	#endif
	#define __close_sock(__fd) close(__fd)
	#define CLEANUP

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
