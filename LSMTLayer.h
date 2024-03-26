#include "ILayer.h"

#pragma once
class LSMTLayer
{
private:
	parameter_t* derivatives_until_memory_deletion = 0;
	size_t trained_steps_since_memory_deletion = 0;
};

