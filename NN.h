#include "ILayer.h"

#pragma once
class NN
{
private:
	ILayer *layers;
	size_t max_layer_count = 0;
	size_t layer_count = 0;

public:
	data_t* Execute(data_t* input, data_t* execution_results);
};

