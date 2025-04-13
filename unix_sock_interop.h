
#include <sys/types.h>
# ifdef _WIN32
	#include <WinSock2.h>
	#include <afunix.h>
	
	#define __close_sock(__fd) closesocket(__fd)
	//#define BIND_PATH "C:\\Users\\Public\\sock"
	#define BIND_PATH "abstract_local_NN"
	#define CLEANUP WSACleanup();

#else
	#include <unistd.h>
	#include <sys/socket.h>
	#include <sys/un.h>
	#include <sys/select.h>

	#define __close_sock(__fd) close(__fd)
	#define bind_PATH "/tmp/NN_socket"
	#define CLEANUP
#endif

#include <stdio.h>

#include "NN_socket_interpreter.h"

int main(int argc, char *argv[]);
int connect();
