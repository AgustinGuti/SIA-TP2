import numpy as np
import math
import yaml
from common import Character, Variables, VARIABLES_ARRAY
from crossovers import crossover, CrossoverConfig
from mutation import mutation, MutationConfig
from selection import selection, SelectionConfig

class AlgorithmConfig:
    def __init__(self, population_size, class_name, selection_config: SelectionConfig, crossover_config: CrossoverConfig, mutation_config: MutationConfig):
        self.population_size = population_size,
        self.class_name = class_name,
        self.selection_config = selection_config,
        self.crossover_config = crossover_config,
        self.mutation_config = mutation_config

def create_population(population_size, class_name):
    population = []
    for _ in range(population_size):
        points_to_assign = 150
        variables = []
        for i in range(len(VARIABLES_ARRAY)-1):
            if i == len(VARIABLES_ARRAY)-2:
                variables.append(points_to_assign)
            else:
                variable = np.random.randint(0, points_to_assign)
                variables.append(variable)
                points_to_assign -= variable

        variables.append(np.random.randint(130, 200)/100)
        population.append(Character(class_name, Variables(*variables)))

    return population

def algorithm_iteration(population, population_to_keep, config: AlgorithmConfig):
    #  Traditional implementation
    new_children = []
    # TODO pueden ser mas o menos hijos y despues cortar menos o mas en la seleccion
    while len(new_children) < population_to_keep:
        parent1 = np.random.choice(population)
        parent2 = np.random.choice(population)
        child1, child2 = crossover(parent1, parent2, config.crossover_config)
        child1 = mutation(child1, config.mutation_config)
        child2 = mutation(child2, config.mutation_config)
        new_children.append(child1)
        new_children.append(child2)

    all_population = population + new_children
    return selection(all_population, config.selection_config)

def algorithm(population_to_keep, iterations, same_best_solution, best_solution_decimals, config: AlgorithmConfig):
    population = create_population(config.population_size, config.class_name)
    with open("log.txt", "w") as file:
        file.write(f"Initial population: {population}, AVG: {np.mean([x.performance for x in population])}\n")

    population.sort(key=lambda x: x.performance, reverse=True)
    best = population[0].performance
    same_iterations = 0
    iteration = 0
    while same_iterations < same_best_solution:
        iteration += 1
        population = algorithm_iteration(population, population_to_keep, config)
        population.sort(key=lambda x: x.performance, reverse=True)
        if round(population[0].performance, best_solution_decimals) == round(best, best_solution_decimals):
            same_iterations += 1
        elif population[0].performance > best:
            best = population[0].performance
            same_iterations = 0
        with open("log.txt", "a") as file:
            file.write(f"Iteration: {iteration}, population: {population[0]}, AVG: {np.mean([x.performance for x in population])}\n") 

    return population

def main():
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    population_to_keep = config["population_to_keep"]
    iterations = config["iterations"]
    same_best_solution = config["same_best_solution"]
    best_solution_decimals = config["best_solution_decimals"]

    selection_config = SelectionConfig(config["selection_config"]["type"], population_to_keep, config["selection_config"]["tournament_size"], config["selection_config"]["threshold"], config["selection_config"]["temperature"])
    crossover_config = CrossoverConfig(config["crossover_config"]["type"])
    mutation_config = MutationConfig(config["mutation_config"]["type"], config["mutation_config"]["mutation_rate"], config["mutation_config"]["delta_mutation"])

    config = AlgorithmConfig(config["population_size"], config["class_name"], selection_config, crossover_config, mutation_config)
    result = algorithm(population_to_keep,  iterations, same_best_solution, best_solution_decimals, config)
    result.sort(key=lambda x: x.performance, reverse=True)
    # print(result)

if __name__ == "__main__":
    main()