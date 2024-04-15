from scipy.optimize import minimize
import numpy as np
from scipy.optimize import LinearConstraint

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

calculations = {
    "attack": lambda strength, agility, expertise, resistence, life, height: (agility + expertise)*strength*modifiers_calculations["ATM"](height),
    "defense": lambda strength, agility, expertise, resistence, life, height: (resistence + expertise)*life*modifiers_calculations["DEM"](height)
}

VARIABLES_ARRAY = ["strength", "agility", "expertise", "resistance", "life", "height"]

def calculate_performance(variables):
    height = variables[-1]
    variables = [stats_calculation[x](y) for x, y in zip(VARIABLES_ARRAY, variables[:-1])]
    variables.append(height)
    attack = calculations["attack"](*variables)

    defense = calculations["defense"](*variables)
    return -classes_performance_calculation["assasin"](attack, defense)

def main():
    constraints = LinearConstraint([[1, 1, 1, 1, 1, 0]], [149.999999999999], [150.000000000001])
    constraint_height = LinearConstraint([[0, 0, 0, 0, 0, 1]], [1.3], [2])
    constraints_positive = LinearConstraint(np.eye(6), [0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = minimize(calculate_performance, x0, method='SLSQP', constraints=[constraints, constraint_height, constraints_positive], options={'ftol': 1e-8, 'disp': True})
    print("Maximum value:", -result.fun) 
    named_result = [{x: y} for x, y in zip(VARIABLES_ARRAY, result.x)]
    print("Location of maximum:", named_result)

if __name__ == "__main__":
    main()