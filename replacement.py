import random
import math

ALLOWED_REPLACEMENTS = ['traditional', 'young_bias', 'generational']

class ReplacementConfig:
    def __init__(self, type, quantity, gen_gap=0):
        if type not in ALLOWED_REPLACEMENTS:
            raise ValueError(f"Invalid type. Valid types are: {ALLOWED_REPLACEMENTS}")
        self.type = type
        self.quantity = quantity
        self.gen_gap = gen_gap

    def __str__(self):
        return f"ReplacementConfig(type={self.type}, gen_gap={self.gen_gap})"
    
    def __repr__(self):
        return str(self)
    
def replacement(population, new_children, config: ReplacementConfig):
    return REPLACEMENT_METHODS[config.type](population, new_children, config)

def _traditional_replacement(population, new_children, config):
    return random.choices(population + new_children, k=config.quantity)

def _young_replacement(population, new_children, config):
    children_qty = len(new_children)
    if children_qty > config.quantity:
        return random.choices(new_children, k=config.quantity)
    else:
        return new_children + random.choices(population, k=config.quantity - children_qty)

def _generational_gap_replacement(population, new_children, config):
    return (random.choices(population, k=math.ceil((1-config.gen_gap)*config.quantity)) + random.choices(new_children, k=math.ceil((config.gen_gap)*config.quantity)))[:config.quantity]
    
REPLACEMENT_METHODS = {
    'traditional': _traditional_replacement,
    'young_bias': _young_replacement,
    'generational': _generational_gap_replacement
}