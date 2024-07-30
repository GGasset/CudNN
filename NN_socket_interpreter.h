#include <vector>
#include <stdio.h>

#include "NN_enums.h"
#include "HashTable.h"
#include "NN_constructor.h"

#pragma once

typedef struct {
	size_t accumulated_training_t_count;
	data_t* accumulated_activations;
	data_t* accumulated_execution_values;
	data_t* accumulated_Y_hat;
	NN* network;
} network_container;

typedef struct {
	data_t *return_value;
	size_t value_count;
	size_t error;
} return_specifier;

class NN_manager
{
private:
	size_t network_count = 0;
	HashTable<size_t, network_container*>* networks = 0;
public:
	NN_manager(size_t bucket_count);

	return_specifier* parse_message(void* message, size_t message_length);
};
