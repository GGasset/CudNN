#include <unistd.h>
#include <cstdio>

#include <sys/socket.h>
#include <sys/un.h>

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
