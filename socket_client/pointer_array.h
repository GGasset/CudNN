#include <stddef.h>

#pragma once
template<typename T>
class pointer_array
{
private:
	int length;
	T* data;
public:
	pointer_array()
	{
		length = 0;
		data = 0;
	}

	void _init(int length)
	{
		if (length <= 0) throw;
		this->length = length;
		data = new T[length];
	}

	~pointer_array()
	{
		delete[] data;
	}

	T* get_raw()
	{
		return data;
	}

	void fill(T value)
	{
		for (size_t i = 0; i < length; i++) data[i] = value;
	}

	int get_length()
	{
		return length;
	}

	T get(int i)
	{
		return data[i];
	}

	void set(T value, int i)
	{
		data[i] = value;
	}

	void cpy(void* dst, size_t dst_copy_start_bytes)
	{
		for (size_t i = 0; i < length; i++) (T*)(dst + dst_copy_start_bytes + sizeof(T) * i) = data[i];
	}
};
