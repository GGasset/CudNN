#ifndef RegisterTypes
#define RegisterTypes

#include <godot_cpp/godot.hpp>
#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>

#include "godot_register_types.h"


using namespace godot;

void NN_constructor::_bind_methods()
{
	//ClassDB::bind_method(D_METHOD("append_layer", "connections_type", "neurons_type", "neuron_count", "activation"), &NN_constructor::append_layer);
	//ClassDB::bind_method(D_METHOD("construct", "input_length", "stateful"), &NN_constructor::construct);
}

void NN::_bind_methods()
{
	/*
	ClassDB::bind_method(D_METHOD("execute", "input"), &NN::inference_execute);

	ClassDB::bind_method(D_METHOD("training_execute", "t_count", "X", "Y", "get_Y", "execution_values", "activations", "old_array_length"), &NN::training_execute);
	ClassDB::bind_method(D_METHOD("train", "t_count", "execution_values", "activations", "Y_hat", "copy_Y_hat_to_gpu", "Y_hat_value_count", "cost_function", "learning_rate", "gradient_clip", "dropout_rate"), &NN::train);

	ClassDB::bind_method(D_METHOD("evolve"), &NN::evolve);
	ClassDB::bind_method(D_METHOD("add_input_neuron"), &NN::add_input_neuron);
	ClassDB::bind_method(D_METHOD("add_output_neuron"), &NN::add_output_neuron);

	ClassDB::bind_method(D_METHOD("delete_memory"), &NN::delete_memory);
	ClassDB::bind_method(D_METHOD("clone"), &NN::delete_memory);
	*/
}

void initialize_example_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	//ClassDB::register_class<NN>();
	//ClassDB::register_class<NN_constructor>();
}

void uninitialize_example_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}

extern "C" {
// Initialization.
GDExtensionBool GDE_EXPORT NN_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_example_module);
	init_obj.register_terminator(uninitialize_example_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}
}
#endif
