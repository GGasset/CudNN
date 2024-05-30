#include "IConnections.h"

void IConnections::generate_random_values(float** pointer, size_t float_count, size_t start_i)
{
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	//curandSetPseudoRandomGeneratorSeed(generator, 15);
	curandGenerateUniform(generator, *pointer + start_i, float_count);
}

void IConnections::deallocate()
{
}