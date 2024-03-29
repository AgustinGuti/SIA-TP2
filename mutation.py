import numpy as np

from common import VARIABLES_ARRAY, Variables, Character, fix_variable_limit

def mutation(character: Character, mutation_rate, delta_mutation, mutation_type):
    if mutation_type == 'gen':
        return _gen_mutation(character, mutation_rate, delta_mutation)
    elif mutation_type == 'multi_gen':
        return _multigen_mutation(character, mutation_rate, delta_mutation)
    else:
        raise ValueError('Invalid mutation type, please choose between "gen" and "multi_gen"')

def _mutate_index(character: Character, delta_mutation, index): 
    if index == len(VARIABLES_ARRAY)-1:
        delta = np.random.randint(-delta_mutation, delta_mutation)/100
        new_value = character.variables.height + delta
        if new_value < 1.3:
            new_value = 1.3
        elif new_value > 2:
            new_value = 2
    else:
        delta = np.random.randint(-delta_mutation, delta_mutation)
        new_value = getattr(character.variables, VARIABLES_ARRAY[index]) + delta
        if new_value < 0:
            new_value = 0
        elif new_value > 150:
            new_value = 150

    new_items_array = [getattr(character.variables, x) for x in VARIABLES_ARRAY]
    new_items_array[index] = new_value

    return new_items_array

def _gen_mutation(character: Character, mutation_rate, delta_mutation):
    if np.random.rand() < mutation_rate:
        mutation_index = np.random.randint(0, len(VARIABLES_ARRAY)-1)
        new_items_array = _mutate_index(character, delta_mutation, mutation_index)
        new_items_array = fix_variable_limit(new_items_array)

        return Character(character.class_name, Variables(*new_items_array))
    return character

def _multigen_mutation(character: Character, mutation_rate, delta_mutation):
    new_items_array = [getattr(character.variables, x) for x in VARIABLES_ARRAY]
    for i in range(len(VARIABLES_ARRAY)):
        if np.random.rand() < mutation_rate:
            mutation_index = i
            new_items_array = _mutate_index(character, delta_mutation, mutation_index)

    new_items_array = fix_variable_limit(new_items_array)
    return Character(character.class_name, Variables(*new_items_array))
