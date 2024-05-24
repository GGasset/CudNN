#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

__device__ data_t device_min(data_t a, data_t b);

