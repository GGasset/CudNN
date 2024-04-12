#include "ILayer.h"

#pragma once
class LSMTLayer : public ILayer
{
public:
	field_t* derivatives_until_memory_deletion = 0;
	size_t trained_steps_since_memory_deletion = 0;

	field_t* neuron_weights = 0;
	data_t* state = 0;

	void execute(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start
	) override;
};

