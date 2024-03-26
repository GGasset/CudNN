#include "ILayer.h"

#pragma once
class NeuronLayer : public ILayer
{
protected:
	ActivationFunctions activation = ActivationFunctions::sigmoid;

public:
	void execute(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start
	) override;
};

