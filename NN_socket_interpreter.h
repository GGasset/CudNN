#include <vector>
#include <stdio.h>

#include "NN_constructor.h"

#pragma once

typedef struct {
	data_t *return_value;
	size_t value_count;
	size_t error;
} return_specifier;

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

	NN_manager();

	return_specifier parse_message(void* message, size_t message_length);
};
