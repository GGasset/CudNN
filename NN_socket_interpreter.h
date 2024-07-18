#include <vector>

#include "NN_constructor.h"

#pragma once

typedef struct return_specifier {
	data_t *return_value;
	size_t value_count;
};

class NN_manager
{
private:
	size_t network_count = 0;
	std::vector<size_t> accumulated_training_t_count;
	std::vector<data_t*> accumulated_activations;
	std::vector<data_t*> accumulated_execution_values;
	std::vector<data_t*> accumulated_Y_hat;
	std::vector<NN*> networks;

public:
	enum action_enum : size_t
	{
		construct = 0,
		last_entry
	};
};
