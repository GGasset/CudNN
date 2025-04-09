
# ifdef _WIN32
	#include <WinSock2.h>
	#include <sys/types.h>
	#include <afunix.h>
	
	#define __close_sock(__fd) closesocket(__fd)
	#define BIND_PATH "C:\\sock"
	#define UNBIND_FILE(PATH)
#else
	#include <unistd.h>
	#include <sys/socket.h>
	#include <sys/un.h>
	#include <sys/select.h>

	#define __close_sock(__fd) close(__fd)
	#define bind_PATH "/tmp/NN_socket"
	#define UNBIND_FILE(PATH) unlink(PATH)
#endif

#include <stdio.h>

#include "NN_socket_interpreter.h"

int main(int argc, char *argv[]);
int connect();
