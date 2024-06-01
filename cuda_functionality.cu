#include "cuda_functionality.cuh"

__device__ data_t device_min(data_t a, data_t b)
{
	return a * (a <= b) + b * (b < a);
}

__device__ size_t get_tid()
{
	return blockIdx.x * blockDim.z + threadIdx.z * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}