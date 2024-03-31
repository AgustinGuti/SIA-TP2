import numpy as np

VARIABLES_ARRAY = ["strength", "agility", "expertise", "resistance", "life", "height"]
MAX_ATTRIBUTE_SUM = 150


classes_performance_calculation = {
    "warrior": lambda x, y: 0.6*x + 0.4*y, 
    "archer": lambda x, y: 0.9 * x + 0.1 * y,
    "defender": lambda x, y: 0.1 * x + 0.9 * y,
    "assasin": lambda x, y: 0.8 * x + 0.3 * y
}

stats_calculation = {
    "strength": lambda x: 100 * np.tanh(0.01*x),
    "agility": lambda x: np.tanh(0.01*x),
    "expertise": lambda x: 0.6 * np.tanh(0.01*x),
    "resistance": lambda x: np.tanh(0.01*x),
    "life": lambda x: 100 * np.tanh(0.01*x),
}

modifiers_calculations = {
    "ATM": lambda x: 0.5 - (3*x-5)**4 + (3*x-5)**2 + x/2,
    "DEM": lambda x: 2+(3*x-5)**4 - (3*x-5)**2 - x/2,
}

class Variables:
    def __init__(self, strength, agility, expertise, resistance, life, height):
        self.strength = strength
        self.agility = agility
        self.expertise = expertise
        self.resistance = resistance
        self.life = life
        self.height = height

    def __str__(self):
        return f"STR:{self.strength:8.2f} AGI:{self.agility:8.2f} EXP:{self.expertise:8.2f} RES:{self.resistance:8.2f} LIF:{self.life:8.2f} HEI:{self.height:8.2f}"

    def __repr__(self):
        return self.__str__()
    
    def equals(self, other, tolerance=0.01):
        return all([abs(getattr(self, x) - getattr(other, x)) < getattr(self, x)*tolerance for x in VARIABLES_ARRAY])


class Character:
    def __init__(self, class_name, variables: Variables):
        self.class_name = class_name
        self.variables = variables

        strength = stats_calculation["strength"](variables.strength)
        agility = stats_calculation["agility"](variables.agility)
        expertise = stats_calculation["expertise"](variables.expertise)
        resistance = stats_calculation["resistance"](variables.resistance)
        life = stats_calculation["life"](variables.life)

        attack = (agility + expertise) * strength * modifiers_calculations["ATM"](variables.height)
        defense = (resistance + expertise) * life * modifiers_calculations["DEM"](variables.height)

        self.performance = classes_performance_calculation[class_name](attack, defense)

    def __str__(self):
        return f"\n{self.class_name} - {self.performance:8.2f} - {self.variables}"

    def __repr__(self):
        return self.__str__()

def fix_variable_limit(items):
    height = items[-1]
    items = items[:-1]
    current_sum = sum(items)
    if current_sum != MAX_ATTRIBUTE_SUM:
        items = [MAX_ATTRIBUTE_SUM * x/current_sum for x in items]

    items.append(height)
    return items