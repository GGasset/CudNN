#include "ILayer.h"

#pragma once
class NeuronLayer : ILayer
{
private:
	ActivationFunctions activation;

public:
	void execute(
		data_t* activations, size_t activations_start,
		data_t* execution_values, size_t execution_values_start
	) override;
};

