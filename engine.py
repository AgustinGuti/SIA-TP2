import numpy as np
import math
import yaml
import random
import os
import json
import itertools
import multiprocessing
import re

from tqdm import tqdm


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
        self.last_iterations = []

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

def calculate_population_diversity(population):
    # SHANNON DIVERSITY INDEX
    # Count the number of individuals in each group
    group_counts = {}
    for individual in population:
        group = tuple(list(round(x, 0) for x in individual.variables.values()[:-1]) + [round(individual.variables.values()[-1], 2)])
        if group not in group_counts:
            group_counts[group] = 0
        group_counts[group] += 1

    # Calculate the total number of individuals
    total_count = sum(group_counts.values())

    # Calculate the SDI
    sdi = 0
    for count in group_counts.values():
        proportion = count / total_count
        sdi -= proportion * np.log(proportion)

    return sdi


def get_next_filename(result_dir):
    filenames = os.listdir(result_dir)
    indices = [int(re.search(r'result_(\d+).json', filename).group(1)) for filename in filenames if re.search(r'result_(\d+).json', filename)]
    max_index = max(indices) if indices and len(indices) > 0 else -1
    return f"{result_dir}/result_{max_index + 1}.json"


def algorithm(population_to_keep, hard_cap, config: AlgorithmConfig, result_dir="./results"):

    population = create_population(config.population_size, config.class_name)

    current_best: Character = max(population, key=lambda x: x.performance)
    algorithm_data = AlgorithmData(population, (current_best, 0))
    best = current_best, algorithm_data.generation

    # iteration_best = (current_best, algorithm_data.generation)
    # iteration_mean = (np.mean([x.performance for x in algorithm_data.population]), algorithm_data.generation)
    # diversity = calculate_population_diversity(algorithm_data.population)
    # algorithm_data.last_iterations.append({"best": iteration_best, "mean": iteration_mean, "diversity": diversity})
    algorithm_data.best = best
    while not should_end(algorithm_data.generation, algorithm_data.population, current_best, config.end_condition_config) and algorithm_data.generation < hard_cap:
        algorithm_data.generation += 1
        algorithm_data.population = algorithm_iteration(algorithm_data, population_to_keep, config)
        current_best = max(algorithm_data.population, key=lambda x: x.performance)
        iteration_best = (current_best, algorithm_data.generation)
        iteration_mean = (np.mean([x.performance for x in algorithm_data.population]), algorithm_data.generation)
        diversity = calculate_population_diversity(algorithm_data.population)
        algorithm_data.last_iterations.append({"best": iteration_best, "mean": iteration_mean, "diversity": diversity})
        if current_best.performance > best[0].performance:
            best = current_best, algorithm_data.generation
            algorithm_data.best = best

    os.makedirs(result_dir, exist_ok=True)
    filename = get_next_filename(result_dir)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    json_data = {
        "results": {
            "best": {
                "solution": algorithm_data.best[0].full_json(),
                "generation": algorithm_data.best[1]
            },
            "last_iterations": [{"best": x["best"][0].json(), "mean": x["mean"][0], "generation": x["best"][1], "diversity": x["diversity"]} for x in algorithm_data.last_iterations]
        },
        "config": config.json()
    }
    with open(filename, "w") as file:
        json.dump(json_data, file, indent=4)

    return population, best

def build_config(config):
    population_to_keep = config['algorithm_config']["population_to_keep"]
    population_size = config['algorithm_config']["population_size"]

    population_to_keep = int(population_to_keep*population_size)

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

def run_repetition(args):
    param_combination, config, params_names, result_dir = args

    for i, param in enumerate(param_combination):
        config[params_names[i][0]][params_names[i][1]] = param
    algorithm_config, population_to_keep, hard_cap = build_config(config)
    result, best = algorithm(population_to_keep, hard_cap,  algorithm_config, result_dir)

def run_analysis(args):
    param_combination, run_config, config, params_names, result_dir = args
    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(run_repetition, [(param_combination, config, params_names, result_dir) for _ in range(run_config['repetitions'])]), total=run_config['repetitions']):
            pass
    # for _ in range(run_config['repetitions']):
    #     run_repetition((param_combination, config, params_names, result_dir))


def verify_results_integrity(result_dir):
    filenames = os.listdir(result_dir)
    for filename in filenames:
        with open(f"{result_dir}/{filename}", "r") as file:
            try:
                json.load(file)
            except json.JSONDecodeError:
                print(f"Invalid JSON file: {filename}")
                os.remove(f"{result_dir}/{filename}")   

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
        population_size = config['algorithm_config']["population_size"]
        # Iterate over classes
        for class_name in run_config["class_names"]:
            config['algorithm_config']["class_name"] = class_name
            # For each param to change
            params_to_combinate = []
            params_names = []
            if run_config['params_to_change']:
                for param in run_config['params_to_change']:
                    params_names.append((param['param_to_change_group'], param['param_to_change_name']))
                    if param['param_value']['numeric']:
                        min_value = param['param_value']['numeric_value']['min_value']*10000
                        max_value = param['param_value']['numeric_value']['max_value']*10000
                        step = param['param_value']['numeric_value']['step']*10000
                        values = range(math.ceil(min_value), math.ceil(max_value)+math.ceil(step), math.ceil(step))                    
                        values = [math.ceil(value/10000) if math.ceil(value/10000) == value/10000 else value/10000 for value in values]
                    else:
                        values = param['param_value']['categorical_values']

                    params_to_combinate.append(values)
            num_combinations = np.prod([len(values) for values in params_to_combinate])
            print(f"Number of combinations: {num_combinations}")
            print(f"Total number of runs: {num_combinations*run_config['repetitions']}")
            
            data_for_run_analysis = [(param_combination, run_config, config, params_names, run_config['output_dir']) for param_combination in itertools.product(*params_to_combinate)]
            # with multiprocessing.Pool() as pool:
            #     for _ in tqdm(pool.imap_unordered(run_analysis, data_for_run_analysis), total=len(data_for_run_analysis)):
            #         pass
            for i, param_combination in enumerate(data_for_run_analysis):
                print(f"Running combination {i+1}/{len(data_for_run_analysis)}")
                run_analysis(param_combination)

    verify_results_integrity(run_config['output_dir'])

              

if __name__ == "__main__":
    main()