#pragma once

typedef struct evolution_metadata {
	float evolution_metadata_field_mutation_chance = .05;
	float evolution_metadata_field_max_mutation = .02;

	float field_max_evolution = .2;
	float layer_addition_probability = .05;
	float neuron_deletion_probability = .1;
	float neuron_addition_probability = .1;
	float layer_distance_from_added_neuron_connection_addition_modifier = .01;
};