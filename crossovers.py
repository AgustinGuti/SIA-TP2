from common import VARIABLES_ARRAY, Variables, Character, fix_variable_limit
import numpy as np

def one_point_crossover(parent1, parent2):
    point = np.random.randint(0, len(VARIABLES_ARRAY)-1)
    new_items_array_1 = []
    new_items_array_2 = []
    for i in range(len(VARIABLES_ARRAY)):
        if i < point:
            new_items_array_1.append(getattr(parent1.variables, VARIABLES_ARRAY[i]))
            new_items_array_2.append(getattr(parent2.variables, VARIABLES_ARRAY[i]))
        else:
            new_items_array_1.append(getattr(parent2.variables, VARIABLES_ARRAY[i]))
            new_items_array_2.append(getattr(parent1.variables, VARIABLES_ARRAY[i]))

    # Fix 150 limit
    new_items_array_1 = fix_variable_limit(new_items_array_1)
    new_items_array_2 = fix_variable_limit(new_items_array_2)

    return Character(parent1.class_name, Variables(*new_items_array_1)), Character(parent2.class_name, Variables(*new_items_array_2))