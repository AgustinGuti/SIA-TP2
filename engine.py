import numpy as np
import math
import yaml
import random
import os
import json

from common import Character, Variables, fix_variable_limit, VARIABLES_ARRAY, MAX_ATTRIBUTE_SUM
from crossovers import crossover, CrossoverConfig
from mutation import mutation, MutationConfig
from selection import selection, SelectionConfig
from replacement import replacement, ReplacementConfig
from end_condition import should_end, EndConditionConfig

class AlgorithmData:
    def __init__(self, population, best):
        self.population = population
        self.best = best
        self.generation = 0
        self.last_bests = []

    def __str__(self):
        return f"AlgorithmData(population={self.population}, best={self.best}, generation={self.generation})"

    def __repr__(self):
        return str(self)

class AlgorithmConfig:
    def __init__(self, population_size, population_to_keep, class_name, selection_config_a: SelectionConfig, selection_config_b: SelectionConfig, selection_config_rate,
                 crossover_config: CrossoverConfig, mutation_config: MutationConfig, replacement_config_a: ReplacementConfig, replacement_config_b: ReplacementConfig,
                 replacement_config_rate, end_condition_config: EndConditionConfig):
        self.population_size = population_size
        self.population_to_keep = population_to_keep
        self.class_name = class_name
        self.selection_config_a = selection_config_a
        self.selection_config_b = selection_config_b
        self.selection_config_rate = selection_config_rate
        self.crossover_config = crossover_config
        self.mutation_config = mutation_config
        self.replacement_config_a = replacement_config_a
        self.replacement_config_b = replacement_config_b
        self.replacement_config_rate = replacement_config_rate
        self.end_condition_config = end_condition_config

    def json(self):
        return {
            "population_size": self.population_size,
            "population_to_keep": self.population_to_keep,
            "class_name": self.class_name,
            "selection_config_a": self.selection_config_a.json(),
            "selection_config_b": self.selection_config_b.json(),
            "selection_config_rate": self.selection_config_rate,
            "crossover_config": self.crossover_config.json(),
            "mutation_config": self.mutation_config.json(),
            "replacement_config_a": self.replacement_config_a.json(),
            "replacement_config_b": self.replacement_config_b.json(),
            "replacement_config_rate": self.replacement_config_rate,
            "end_condition_config": self.end_condition_config.json()
        }

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

def algorithm_iteration(algorithm_data: AlgorithmData, population_to_keep, config: AlgorithmConfig):
    selected_population_a = selection(algorithm_data.population, algorithm_data.generation, config.selection_config_a)
    selected_population_b = selection(algorithm_data.population, algorithm_data.generation, config.selection_config_b)
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

    replacement_a = replacement(algorithm_data.population, new_children, config.replacement_config_a)
    replacement_b = replacement(algorithm_data.population, new_children, config.replacement_config_b)

    return replacement_a + replacement_b

def algorithm(population_to_keep, hard_cap, config: AlgorithmConfig):
    population = create_population(config.population_size, config.class_name)

    current_best: Character = max(population, key=lambda x: x.performance)
    algorithm_data = AlgorithmData(population, (current_best, 0))
    best = current_best, algorithm_data.generation
    algorithm_data.last_bests.append((current_best, algorithm_data.generation))
    algorithm_data.best = best
    while not should_end(algorithm_data.generation, algorithm_data.population, current_best, config.end_condition_config) and algorithm_data.generation < hard_cap:
        algorithm_data.generation += 1
        algorithm_data.population = algorithm_iteration(algorithm_data, population_to_keep, config)
        current_best = max(algorithm_data.population, key=lambda x: x.performance)
        algorithm_data.last_bests.append((current_best, algorithm_data.generation))
        if current_best.performance > best[0].performance:
            best = current_best, algorithm_data.generation
            algorithm_data.best = best

    i = 0
    filename = f"./results/result_{i}.json"
    while os.path.exists(filename):
        i += 1
        filename = f"./results/result_{i}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    json_data = {
        "results": {
            "best": {
                "solution": algorithm_data.best[0].json(),
                "generation": algorithm_data.best[1]
            },
            "last_bests": [{"character":x[0].json(), "generation": x[1]} for x in algorithm_data.last_bests],
            "last_generation": algorithm_data.generation
        },
        "config": config.json()
    }
    with open(filename, "w") as file:
        json.dump(json_data, file, indent=4)

    return population, best

def build_config(config):
    population_to_keep = config['algorithm_config']["population_to_keep"]
    population_size = config['algorithm_config']["population_size"]

    first_selection = round(config["selection_config"]["ratio_A"]*population_to_keep)
    second_selection = population_to_keep - first_selection
    selection_config_a = SelectionConfig(config["selection_config"]["type_1"], first_selection, config["selection_config"]["tournament_size"], config["selection_config"]["tournament_threshold"], config["selection_config"]["initial_temperature"], config["selection_config"]["temperature_decay"], config["selection_config"]["min_temperature"])
    selection_config_b = SelectionConfig(config["selection_config"]["type_2"], second_selection, config["selection_config"]["tournament_size"], config["selection_config"]["tournament_threshold"], config["selection_config"]["initial_temperature"], config["selection_config"]["temperature_decay"], config["selection_config"]["min_temperature"])
    crossover_config = CrossoverConfig(config["crossover_config"]["type"])
    mutation_config = MutationConfig(config["mutation_config"]["type"], config["mutation_config"]["rate"], config["mutation_config"]["delta"])

    first_replacement = round(config["replacement_config"]["ratio_B"]*population_to_keep)
    second_replacement = population_to_keep - first_replacement
    replacement_config_a = ReplacementConfig(config["replacement_config"]["type_1"], first_replacement, config["replacement_config"]["gen_gap"])
    replacement_config_b = ReplacementConfig(config["replacement_config"]["type_2"], second_replacement, config["replacement_config"]["gen_gap"])
    end_condition_config = EndConditionConfig(config["end_condition_config"]["type"], config["end_condition_config"]["generations_to_check"], config["end_condition_config"]["optimum"], config["end_condition_config"]["tolerance"], config["end_condition_config"]["structure_tolerance"], config["end_condition_config"]["structure_relevant_proportion"], config["end_condition_config"]["generations"])

    algorithm_config = AlgorithmConfig(population_size, population_to_keep, config['algorithm_config']["class_name"], selection_config_a, selection_config_b, config["selection_config"]["ratio_A"],
                                        crossover_config, mutation_config, replacement_config_a, replacement_config_b, config["replacement_config"]["ratio_B"], end_condition_config)
    return algorithm_config, population_to_keep, config["algorithm_config"]["hard_cap_iterations"]

def main():
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open("run_config.yaml", "r") as file:
        run_config = yaml.load(file, Loader=yaml.FullLoader)

    if(run_config["use_default"]):
        algorithm_config, population_to_keep, hard_cap = build_config(config)
        result, best = algorithm(population_to_keep, hard_cap,  algorithm_config)
        print(f"Best solution: {best[0]}")
    else:
        if len(run_config["class_names"]) == 0:
            raise ValueError("Invalid class name")
        for class_name in run_config["class_names"]:
            config['algorithm_config']["class_name"] = class_name
            for _ in range(run_config['repetitions']):
                if run_config['param_value']['numeric']:
                    min_value = run_config['param_value']['numeric_value']['min_value']
                    max_value = run_config['param_value']['numeric_value']['max_value']
                    step = run_config['param_value']['numeric_value']['step']
                    for i in range(min_value, max_value+step, step):
                        print(f"Iteration {i}")
                        config[run_config['param_to_change_group']][run_config['param_to_change_name']] = i
                        algorithm_config, population_to_keep, hard_cap = build_config(config)
                        result, best = algorithm(population_to_keep, hard_cap,  algorithm_config)
                elif run_config['param_value']['categorical']:
                    for i in run_config['param_value']['categorical_values']:
                        config[run_config['param_to_change_group']][run_config['param_to_change_name']] = i
                        algorithm_config, population_to_keep, hard_cap = build_config(config)
                        result, best = algorithm(population_to_keep, hard_cap,  algorithm_config)
                        print(f"Iteration {i} finished")
                else:
                    raise ValueError("Invalid run_config")


if __name__ == "__main__":
    main()