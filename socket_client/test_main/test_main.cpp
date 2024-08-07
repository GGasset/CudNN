#include "../NN_client.h"

int main()
{
	NN_client test_client = NN_client();

	NN_constructor_client constructor = NN_constructor_client();

	constructor.create_minimal_recurrent_NN(1, true, 1);

	test_client.link_NN(constructor);
}
