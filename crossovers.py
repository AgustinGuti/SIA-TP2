from common import VARIABLES_ARRAY, Variables, Character, fix_variable_limit
import numpy as np

class CrossoverConfig:
    def __init__(self, method):
        if method not in condition_functions.keys():
            raise ValueError(f"Invalid method. Valid methods are: {condition_functions.keys()}")
        self.method = method

UNIFORM_PROBABILTY = 0.5

def calculate_point():
    point = np.random.randint(0, len(VARIABLES_ARRAY)-1)
    return point

condition_functions = {
    "one_point": lambda x, point1, point2, length: x < point1,
    "two_point": lambda x, point1, point2, length: x < point1 or x > point2,
    # TODO check if the probabilty must be in config
    "uniform": lambda x, point1, point2, length: np.random.rand() < UNIFORM_PROBABILTY,
    "annular": lambda x, point1, point2, length: not (x < point1 - (len(VARIABLES_ARRAY) - length) or x >= point1)
}

def crossover(parent1, parent2, config: CrossoverConfig):
    if config.method not in condition_functions.keys():
        raise ValueError(f"Invalid method. Valid methods are: {condition_functions.keys()}")
    point1 = np.random.randint(0, len(VARIABLES_ARRAY)-1)
    point2 = np.random.randint(0, len(VARIABLES_ARRAY)-1)
    if point1 > point2:
        point1, point2 = point2, point1
    length = np.random.randint(0, len(VARIABLES_ARRAY)//2)

    new_items_array_1 = []
    new_items_array_2 = []
    for i in range(len(VARIABLES_ARRAY)):
        if condition_functions[config.method](i, point1, point2, length):
            new_items_array_1.append(getattr(parent1.variables, VARIABLES_ARRAY[i]))
            new_items_array_2.append(getattr(parent2.variables, VARIABLES_ARRAY[i]))
        else:
            new_items_array_1.append(getattr(parent2.variables, VARIABLES_ARRAY[i]))
            new_items_array_2.append(getattr(parent1.variables, VARIABLES_ARRAY[i]))    
    # Fix 150 limit
    new_items_array_1 = fix_variable_limit(new_items_array_1)
    new_items_array_2 = fix_variable_limit(new_items_array_2)

    return Character(parent1.class_name, Variables(*new_items_array_1)), Character(parent2.class_name, Variables(*new_items_array_2))