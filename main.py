import numpy as np
import math
import yaml
import random
from common import Character, Variables, fix_variable_limit, VARIABLES_ARRAY, MAX_ATTRIBUTE_SUM
from crossovers import crossover, CrossoverConfig
from mutation import mutation, MutationConfig
from selection import selection, SelectionConfig
from replacement import replacement, ReplacementConfig
from end_condition import should_end, EndConditionConfig


class AlgorithmConfig:
    def __init__(self, population_size, class_name, selection_config: SelectionConfig, 
                 crossover_config: CrossoverConfig, mutation_config: MutationConfig, replacement_config: ReplacementConfig, end_condition_config: EndConditionConfig):
        self.population_size = population_size
        self.class_name = class_name
        self.selection_config = selection_config
        self.crossover_config = crossover_config
        self.mutation_config = mutation_config
        self.replacement_config = replacement_config
        self.end_condition_config = end_condition_config

def create_population(population_size, class_name):
    population = []
    for _ in range(population_size):
        variables = []
        for i in range(len(VARIABLES_ARRAY)-1):
            variables.append(np.random.randint(0, MAX_ATTRIBUTE_SUM))
        variables.append(np.random.randint(130, 200)/100)

        # Normalize values
        variables = fix_variable_limit(variables)
        population.append(Character(class_name, Variables(*variables)))

    return population

def algorithm_iteration(population, population_to_keep, generation, config: AlgorithmConfig):
    selected_population = selection(population, generation, config.selection_config)
    random.shuffle(selected_population)
    new_children = []
    while len(new_children) < population_to_keep and len(selected_population) > 1:
        parent1 = selected_population.pop()
        parent2 = selected_population.pop()
        child1, child2 = crossover(parent1, parent2, config.crossover_config)
        child1 = mutation(child1, config.mutation_config)
        child2 = mutation(child2, config.mutation_config)
        new_children.append(child1)
        new_children.append(child2)

    return replacement(population, new_children, config.replacement_config)

def algorithm(population_to_keep: list[Character], config: AlgorithmConfig):
    population = create_population(config.population_size, config.class_name)
    with open("log.txt", "w") as file:
        file.write(f"Initial population AVG: {np.mean([x.performance for x in population])}\n")

    current_best: Character = max(population, key=lambda x: x.performance)
    generation = 0
    best = current_best, generation
    while not should_end(generation, current_best, config.end_condition_config):
        generation += 1
        population = algorithm_iteration(population, population_to_keep, generation, config)
        current_best = max(population, key=lambda x: x.performance)
        if current_best.performance > best[0].performance:
            best = current_best, generation
        with open("log.txt", "a") as file:
            file.write(f"Iteration: {generation}, best: {current_best}, BEST: {current_best.performance:.2f}, AVG: {np.mean([x.performance for x in population]):.2f}, HSTD: {np.std([x.variables.height for x in population])} \n")  
    return population, best

def main():
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    population_to_keep = config['algorithm_config']["population_to_keep"]

    selection_config = SelectionConfig(config["selection_config"]["type"], population_to_keep, config["selection_config"]["tournament_size"], config["selection_config"]["tournament_threshold"], config["selection_config"]["initial_temperature"], config["selection_config"]["temperature_decay"], config["selection_config"]["min_temperature"])
    crossover_config = CrossoverConfig(config["crossover_config"]["type"])
    mutation_config = MutationConfig(config["mutation_config"]["type"], config["mutation_config"]["rate"], config["mutation_config"]["delta"])
    replacement_config = ReplacementConfig(config["replacement_config"]["type"], config["replacement_config"]["gen_gap"])
    end_condition_config = EndConditionConfig(config["end_condition_config"]["type"], config["end_condition_config"]["generations_to_check"], config["end_condition_config"]["optimum"], config["end_condition_config"]["tolerance"], config["end_condition_config"]["generations"])

    algorithm_config = AlgorithmConfig(config["algorithm_config"]["population_size"], config['algorithm_config']["class_name"], selection_config, crossover_config, mutation_config, replacement_config, end_condition_config)
    result, best = algorithm(population_to_keep, algorithm_config)
    print(f"Best solution: {best[0]} at generation {best[1]}")

if __name__ == "__main__":
    main()