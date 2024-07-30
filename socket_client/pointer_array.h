#pragma once
template<typename T>
class pointer_array
{
pivate:
	int length;
	T* data;
public:
	pointer_array(int length);
	~pointer_array();

	T* get_raw();
	void fill(T value);
	int get_length();
	T get(int i);
	void set(T value, int i);
}
