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
    def __init__(self, population_size, class_name, selection_config_a: SelectionConfig, selection_config_b: SelectionConfig,
                 crossover_config: CrossoverConfig, mutation_config: MutationConfig, replacement_config_a: ReplacementConfig, replacement_config_b: ReplacementConfig,
                 end_condition_config: EndConditionConfig):
        self.population_size = population_size
        self.class_name = class_name
        self.selection_config_a = selection_config_a
        self.selection_config_b = selection_config_b
        self.crossover_config = crossover_config
        self.mutation_config = mutation_config
        self.replacement_config_a = replacement_config_a
        self.replacement_config_b = replacement_config_b
        self.end_condition_config = end_condition_config

    def __str__(self):
        return f"AlgorithmConfig(population_size={self.population_size}, class_name={self.class_name}, selection_config_a={self.selection_config_a}, selection_config_b={self.selection_config_b}, crossover_config={self.crossover_config}, mutation_config={self.mutation_config}, replacement_config_a={self.replacement_config_a}, replacement_config_b={self.replacement_config_b}, end_condition_config={self.end_condition_config})"

    def __repr__(self):
        return str(self)

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
    selected_population_a = selection(population, generation, config.selection_config_a)
    selected_population_b = selection(population, generation, config.selection_config_b)
    selected_population = selected_population_a + selected_population_b
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

    replacement_a = replacement(population, new_children, config.replacement_config_a)
    replacement_b = replacement(population, new_children, config.replacement_config_b)

    return replacement_a + replacement_b

def algorithm(population_to_keep, hard_cap, config: AlgorithmConfig):
    population = create_population(config.population_size, config.class_name)
    with open("log.txt", "w") as file:
        file.write(f"Initial population AVG: {np.mean([x.performance for x in population])}\n")

    current_best: Character = max(population, key=lambda x: x.performance)
    generation = 0
    best = current_best, generation
    while not should_end(generation, population, current_best, config.end_condition_config) and generation < hard_cap:
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
    population_size = config['algorithm_config']["population_size"]

    first_selection = round(config["selection_config"]["ratio_A"]*population_size)
    second_selection = population_size - first_selection
    selection_config_a = SelectionConfig(config["selection_config"]["type_1"], first_selection, config["selection_config"]["tournament_size"], config["selection_config"]["tournament_threshold"], config["selection_config"]["initial_temperature"], config["selection_config"]["temperature_decay"], config["selection_config"]["min_temperature"])
    selection_config_b = SelectionConfig(config["selection_config"]["type_2"], second_selection, config["selection_config"]["tournament_size"], config["selection_config"]["tournament_threshold"], config["selection_config"]["initial_temperature"], config["selection_config"]["temperature_decay"], config["selection_config"]["min_temperature"])
    crossover_config = CrossoverConfig(config["crossover_config"]["type"])
    mutation_config = MutationConfig(config["mutation_config"]["type"], config["mutation_config"]["rate"], config["mutation_config"]["delta"])

    first_replacement = round(config["replacement_config"]["ratio_B"]*population_size)
    second_replacement = population_size - first_replacement
    replacement_config_a = ReplacementConfig(config["replacement_config"]["type_1"], first_replacement, config["replacement_config"]["gen_gap"])
    replacement_config_b = ReplacementConfig(config["replacement_config"]["type_2"], second_replacement, config["replacement_config"]["gen_gap"])
    end_condition_config = EndConditionConfig(config["end_condition_config"]["type"], config["end_condition_config"]["generations_to_check"], config["end_condition_config"]["optimum"], config["end_condition_config"]["tolerance"], config["end_condition_config"]["structure_tolerance"], config["end_condition_config"]["structure_relevant_proportion"], config["end_condition_config"]["generations"])

    algorithm_config = AlgorithmConfig(population_size, config['algorithm_config']["class_name"], selection_config_a, selection_config_b, crossover_config, mutation_config, replacement_config_a, replacement_config_b, end_condition_config)
    result, best = algorithm(population_to_keep,config["algorithm_config"]["hard_cap_iterations"],  algorithm_config)
    print(f"Best solution: {best[0]} at generation {best[1]}")

if __name__ == "__main__":
    main()