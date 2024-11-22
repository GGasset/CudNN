#pragma once
#include <cstdlib>
#include <time.h>

template<typename t>
t max(t a, t b)
{
	return a * (a >= b) + b * (a < b);
}

template<typename t>
t min(t a, t b)
{
	return a * (a <= b) + b * (a > b);
}

unsigned long long get_arbitrary_number();

float get_random_float();
