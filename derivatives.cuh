#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

__global__ void LSMT_derivative_calculation(
	data_t* derivatives, size_t derivatives_start, size_t derivatives_per_neuron, size_t neuron_count,
	data_t* execution_values, size_t execution_values_start, size_t execution_values_layer_start, size_t execution_values_per_neuron,
	field_t* neuron_weights
);