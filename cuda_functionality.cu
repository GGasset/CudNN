#include "cuda_functionality.cuh"

__device__ data_t device_min(data_t a, data_t b)
{
	return a * (a <= b) + b * (b < a);
}

__device__ size_t get_tid()
{
	return blockIdx.x * blockDim.z + threadIdx.z * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__global__ void get_occurrences(size_t array_value_count, size_t* read_array, size_t search_value, size_t* count)
{
	size_t tid = get_tid();
	if (tid > array_value_count) return;

	atomicAdd(count, read_array[tid] == search_value);
}
