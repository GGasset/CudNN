#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

#pragma once
class IConnections
{
public:
	virtual void linear_function(size_t activations_start, data_t* activations,
		data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t layer_execution_values_per_neuron,
		parameter_t* weights, parameter_t* biases, size_t layer_length) = 0;

	virtual void deallocate()
	{

	}
};

