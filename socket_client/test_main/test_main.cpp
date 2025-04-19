#include "../NN_client.h"

int main()
{
#ifdef _WIN32
	WSADATA details{};
	if (WSAStartup(MAKEWORD(2, 2), &details)) throw;
#endif

	NN_client test_client = NN_client();

	NN_constructor_client constructor = NN_constructor_client();

	constructor.create_minimal_recurrent_NN(1, true, 1);

	test_client.link_NN(constructor);
	test_client.~NN_client();
	CLEANUP
}
