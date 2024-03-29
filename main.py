import numpy as np
import math
import yaml
from common import Character, Variables, VARIABLES_ARRAY
from crossovers import crossover
from mutation import mutation
from selection import *

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

def algorithm_iteration(population, population_to_keep, delta_mutation, mutation_rate, mutation_type, crossover_type):
    #  Traditional implementation
    population_size = len(population)
    new_children = []
    # TODO pueden ser mas o menos hijos y despues cortar menos o mas en la seleccion
    while len(new_children) < population_size:
        parent1 = np.random.choice(population)
        parent2 = np.random.choice(population)
        child1, child2 = crossover(parent1, parent2, crossover_type)
        child1 = mutation(child1, mutation_rate, delta_mutation, mutation_type)
        child2 = mutation(child2, mutation_rate, delta_mutation, mutation_type)
        new_children.append(child1)
        new_children.append(child2)

    all_population = population + new_children
    new_population = elite_selection(all_population, population_size)

    return new_population

def algorithm(population_size, population_to_keep, class_name, iterations, delta_mutation, mutation_rate, mutation_type, crossover_type, same_best_solution, best_solution_decimals):
    population = create_population(population_size, class_name)
    with open("log.txt", "w") as file:
        file.write(f"Initial population: {population}, AVG: {np.mean([x.performance for x in population])}\n")

    population.sort(key=lambda x: x.performance, reverse=True)
    best = population[0].performance
    same_iterations = 0
    iteration = 0
    while same_iterations < same_best_solution:
        iteration += 1
        population = algorithm_iteration(population, population_to_keep, delta_mutation, mutation_rate, mutation_type, crossover_type)
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

    population_size = config["population_size"]
    population_to_keep = config["population_to_keep"]
    class_name = config["class_name"]
    iterations = config["iterations"]
    delta_mutation = config["delta_mutation"]
    mutation_rate = config["mutation_rate"]
    same_best_solution = config["same_best_solution"]
    best_solution_decimals = config["best_solution_decimals"]
    mutation_type = config["mutation_type"]
    crossover_type = config["crossover_type"]
    temperature = config["temperature"]
    result = algorithm(population_size, population_to_keep, class_name, iterations, delta_mutation, mutation_rate,mutation_type, crossover_type, same_best_solution, best_solution_decimals)
    result.sort(key=lambda x: x.performance, reverse=True)
    # print(result)

if __name__ == "__main__":
    main()