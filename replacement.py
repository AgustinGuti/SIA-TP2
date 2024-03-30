import random
import math

TYPES = ['traditional', 'young_bias', 'generational']

class ReplacementConfig:
    def __init__(self, type, gen_gap):
        if type not in TYPES:
            raise ValueError(f"Invalid type. Valid types are: {TYPES}")
        self.type = type
        self.gen_gap = gen_gap
    
def replacement(population, new_children, config: ReplacementConfig):
    return REPLACEMENT_METHODS[config.type](population, new_children, config)

def _traditional_replacement(population, new_children, config):
    return random.choices(population + new_children, k=len(population))

def _young_replacement(population, new_children, config):
    population_size = len(population)
    children_qty = len(new_children)
    if children_qty > population_size:
        return random.choices(new_children, k=population_size)
    else:
        return new_children + random.choices(population, k=population_size - children_qty)

def _generational_gap_replacement(population, new_children, config):
    population_size = len(population)
    return random.choices(population, k=math.floor((1-config.gen_gap)*population_size)) + random.choices(new_children, k=math.floor((config.gen_gap)*population_size))
    
REPLACEMENT_METHODS = {
    'traditional': _traditional_replacement,
    'young_bias': _young_replacement,
    'generational': _generational_gap_replacement
}