import math
import numpy as np
import random

class SelectionConfig:
    def __init__(self, selection_type, population_to_keep, tournament_size=None, threshold=None, initial_temperature=None, temperature_decay=None, min_temperature=None):
        self.selection_type = selection_type
        self.population_to_keep = population_to_keep
        self.tournament_size = tournament_size
        self.threshold = threshold
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature

def selection(population, generation, config: SelectionConfig):
    return SELECTION_METHODS[config.selection_type](population, generation, config)

def _elite_selection(population, generation, config: SelectionConfig):
    population.sort(key=lambda x: x.performance, reverse=True)
    population_size = len(population)
    new_population = []
    for i in range(0, len(population)-1):
        n = math.ceil((config.population_to_keep-i)/population_size)
        new_population.extend([population[i]]*n)
    
    return new_population

def _roulette_selection(population, generation, config: SelectionConfig):
    return _base_selection(population, config.population_to_keep, config.selection_type)

def _universal_selection(population, generation, config: SelectionConfig):
    return _base_selection(population, config.population_to_keep, 'universal')

def _boltzmann_selection(population, generation, config: SelectionConfig):
    return _base_selection(population, config.population_to_keep, 'bolztmann', config.initial_temperature, config.temperature_decay, config.min_temperature, generation)

def _deterministic_tournament_selection(population, config: SelectionConfig):
    new_population = []
    while len(new_population) < config.population_to_keep:
        tournament = random.choices(population, k=config.tournament_size)
        tournament.sort(key=lambda x: x.performance, reverse=True)
        new_population.append(tournament[0])
    return new_population

def _probabilistic_tournament_selection(population, config: SelectionConfig):
    new_population = []
    while len(new_population) < config.population_to_keep:
        pair = random.choices(population, k=2)
        pair.sort(key=lambda x: x.performance, reverse=True)
        if np.random.rand() < config.threshold:
            new_population.append(pair[0])
        else:
            new_population.append(pair[1])
        
    return new_population

def _get_aptitude(population, individual_index, method, temperature, total_performance, average_performance):
    if method == 'linear':
        return population[individual_index].performance/total_performance
    elif method == 'boltzmann':
        return math.exp(population[individual_index].performance/temperature)/average_performance
    elif method == 'ranking':
        rank = population.index(population[individual_index])
        population_len = len(population)
        return (population_len - rank)/population_len


random_type_by_selection = {
    'roulette': 'roulette',
    'universal': 'universal',
    'bolztmann': 'roulette',
    'ranking': 'roulette'
}

aptitude_type_by_selection = {
    'roulette': 'linear',
    'universal': 'linear',
    'bolztmann': 'boltzmann',
    'ranking': 'ranking'
}

def _base_selection(population, population_to_keep, selection_type, initial_temperature=0, temperature_decay=0, min_temperature=0, generation=None):
    population_size = len(population)
    total_performance = sum([x.performance for x in population])
    average_performance = np.average([x.performance for x in population])
    relative_performances = []
    accumulated_relative_performances = []
    randoms = _calculate_randoms(population_size, population_to_keep, random_type_by_selection[selection_type])
    temperature = 0
    if selection_type == 'ranking': 
        population.sort(key=lambda x: x.performance, reverse=True)
    elif selection_type == 'bolztmann':
        temperature = min_temperature + (initial_temperature - min_temperature) * math.exp(-temperature_decay*generation)
    for i in range(0, population_size):
        new_performance = _get_aptitude(population, i, aptitude_type_by_selection[selection_type], temperature, total_performance, average_performance)
        relative_performances.append(new_performance)
        if i == 0:
            accumulated_relative_performances.append(new_performance)
        else:
            accumulated_relative_performances.append(accumulated_relative_performances[i-1]+new_performance)

    accum_index = 0
    new_population = []

    index = 0
    accumulated_relative_performances_len = len(accumulated_relative_performances)
    while index < accumulated_relative_performances_len and randoms[index] < accumulated_relative_performances[accum_index]:
        new_population.append(population[accum_index])
        accum_index += 1
        index += 1
    
    randoms = randoms[index:]
    for rand in randoms:
        assigned = False
        while not assigned:
            if accumulated_relative_performances[accum_index-1] < rand <= accumulated_relative_performances[accum_index]:
                assigned = True
                new_population.append(population[accum_index])
            else:
                accum_index += 1

    while len(new_population) < population_to_keep:
        new_population.append(population[-1])

    return new_population

def _calculate_randoms(population_size, population_to_keep, selection_type):
    randoms = []
    if selection_type == 'roulette':
        randoms = [np.random.rand() for i in range(0, population_size)]
    elif selection_type == 'universal':
        randoms = [(np.random.rand()+i)/population_to_keep for i in range(0, population_to_keep-1)]
    else:
        raise ValueError('Invalid selection type')
    randoms.sort()
    return randoms

SELECTION_METHODS = {
    'elite': _elite_selection,
    'roulette': _roulette_selection,
    'ranking': _roulette_selection,
    'universal': _universal_selection,
    'boltzmann': _boltzmann_selection,
    'deterministic_tournament': _deterministic_tournament_selection,
    'probabilistic_tournament': _probabilistic_tournament_selection
}