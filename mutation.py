import numpy as np

from common import VARIABLES_ARRAY, Variables, Character, fix_variable_limit

def gen_mutation(character: Character, delta_mutation):
    mutation_index = np.random.randint(0, len(VARIABLES_ARRAY)-1)
    if mutation_index == len(VARIABLES_ARRAY)-1:
        delta = np.random.randint(-delta_mutation, delta_mutation)/100
        new_value = character.variables.height + delta
        if new_value < 1.3:
            new_value = 1.3
        elif new_value > 2:
            new_value = 2
    else:
        delta = np.random.randint(-delta_mutation, delta_mutation)
        new_value = getattr(character.variables, VARIABLES_ARRAY[mutation_index]) + delta
        if new_value < 0:
            new_value = 0
        elif new_value > 150:
            new_value = 150

    new_items_array = [getattr(character.variables, x) for x in VARIABLES_ARRAY]
    new_items_array = fix_variable_limit(new_items_array)

    return Character(character.class_name, Variables(*new_items_array))