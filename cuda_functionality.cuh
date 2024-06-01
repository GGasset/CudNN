#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "data_type.h"

__device__ data_t device_min(data_t a, data_t b);

/// <summary>
/// Calculates linear thread_id up to blockIdx.x [inclusive]
/// </summary>
/// <returns></returns>
__device__ size_t get_tid();

__global__ void get_occurrences(size_t array_value_count, size_t* read_array, size_t search_value, size_t* count);

