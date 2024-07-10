#ifndef RegisterTypes
#define RegisterTypes

#include <godot_cpp/godot.hpp>
#include <godot_cpp/core/class_db.hpp>
#include "../NN_constructor.h"
#include "../NN.h"


using namespace godot;

void NN_constructor::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("append_layer", "connections_type", "neurons_type", "neuron_count", "activation"), &NN_constructor::append_layer);
	ClassDB::bind_method(D_METHOD("construct", "input_length", "stateful"), &NN_constructor::construct);
}

void NN::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("execute", "input"), &NN::execute)

	ClassDB::bind_method(D_METHOD("training_execute", "t_count", "X", "Y", "get_Y", "execution_values", "activations", "old_array_length"), &NN::training_execute);
	ClassDB::bind_method(D_METHOD("train", "t_count", "execution_values", "activations", "Y_hat", "copy_Y_hat_to_gpu", "Y_hat_value_count", "cost_function", "learning_rate", "gradient_clip", "dropout_rate"), &NN::train);

	ClassDB::bind_method(D_METHOD("evolve"), &NN::evolve);
	ClassDB::bind_method(D_METHOD("add_input_neuron"), &NN::add_input_neuron);
	ClassDB::bind_method(D_METHOD("add_output_neuron"), &NN::add_output_neuron);

	ClassDB::bind_method(D_METHOD("delete_memory"), &NN::delete_memory);
	ClassDB::bind_method(D_METHOD("clone"), &NN::delete_memory);
}

void register_gameplay_types(godot::ModuleInitializationLevel p_level) {
  if (p_level != godot::ModuleInitializationLevel::MODULE_INITIALIZATION_LEVEL_SCENE) {
	return;
  }

  // REGISTER CLASSES

  ClassDB::register_class<NN>();
  ClassDB::register_class<NN_constructor>();
}

void unregister_gameplay_types(godot::ModuleInitializationLevel p_level) {
  // DO NOTHING
}

extern "C" {
	GDExtensionBool GDE_EXPORT gameplay_library_init(const GDExtensionInterface *p_interface, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
		  godot::GDExtensionBinding::InitObject init_object(p_interface, p_library, r_initialization);

		  init_object.register_initializer(register_gameplay_types);
		  init_object.register_terminator(unregister_gameplay_types);
		  init_object.set_minimum_library_initialization_level(godot::ModuleInitializationLevel::MODULE_INITIALIZATION_LEVEL_SCENE);

		  return init_object.init();
	}
}

#endif
