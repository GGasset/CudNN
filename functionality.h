#pragma once
#include <cstdlib>

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

float get_random_float();
