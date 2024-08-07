#include <stddef.h>

#include "data_type.h"
#include "NN_enums.h"

#include "pointer_array.h"
#include "client.h"
#include "NN_constructor_client.h"

class NN_client
{
	data_t network_id = -1;
	socket_client socket;

public:
	NN_client();
	~NN_client();

	void link_NN(NN_constructor_client constructor);

private:
	void disconnect_NN();
};
