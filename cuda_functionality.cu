#include "cuda_functionality.cuh"

__device__ data_t device_min(data_t a, data_t b)
{
	return a * (a <= b) + b * (b < a);
}

__device__ size_t get_tid()
{
	return blockIdx.x * blockDim.z + threadIdx.z * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__global__ void mutate_field_array(
	field_t* array, size_t length, 
	float mutation_chance, float max_mutation, 
	float* normalized_random_arr0, float* normalized_random_arr1, float* normalized_random_arr2
)
{
	size_t tid = get_tid();
	if (tid > length) return;

	array[tid] += normalized_random_arr0[tid] * max_mutation * (normalized_random_arr1[tid] < mutation_chance);
	array[tid] *= 1 - 2 * (normalized_random_arr2[tid] < .5);
}