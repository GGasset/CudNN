#include "pointer_array.h"

pointer_array::pointer_array(int length)
{
	if (length <= 0) throw;
	this->length = length;
	data = new T[length];
}

pointer_array::~pointer_array()
{
	delete[] data;
}

T* pointer_array::get_raw()
{
	return data;
}

void pointer_array::fill(T value)
{
	for (size_t i = 0; i < length; i++) data[i] = value;
}

int pointer_array::get_length()
{
	return length;
}

T pointer_array::get(int i)
{
	return data[i];
}

void pointer_array::set(T value, int i)
{
	data[i] = value;
}
