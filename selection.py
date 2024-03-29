import math
import numpy as np

def elite_selection(population, population_to_keep):
    population.sort(key=lambda x: x.performance, reverse=True)
    population_size = len(population)
    new_population = []
    for i in range(0, len(population)-1):
        n = math.ceil((population_to_keep-i)/population_size)
        new_population.extend([population[i]]*n)
    
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

def roulette_selection(population, population_to_keep):
    return _base_selection(population, population_to_keep, 'roulette')

def universal_selection(population, population_to_keep):
    return _base_selection(population, population_to_keep, 'universal')

def boltzmann_selection(population, population_to_keep, temperature):
    return _base_selection(population, population_to_keep, 'roulette', True, temperature)

def _get_aptitude(population, individual_index, method, temperature, total_performance, average_performance):
    if method == 'linear':
        # TODO divide by total performance
        return population[individual_index].performance/total_performance
    elif method == 'boltzmann':
        # TODO this is not used
        return math.exp(population[individual_index].performance/temperature)/average_performance
    

def _base_selection(population, population_to_keep, selection_type, is_boltzmann=False, temperature=0):
    population_size = len(population)
    total_performance = sum([x.performance for x in population])
    average_performance = np.average([x.performance for x in population])
    relative_performances = []
    accumulated_relative_performances = []
    randoms = _calculate_randoms(population_size, population_to_keep, selection_type)
    for i in range(0, population_size-1):
        aptitude_name = 'boltzmann' if is_boltzmann else 'linear'
        new_performance = _get_aptitude(population, i, aptitude_name, temperature, total_performance, average_performance)
        relative_performances.append(new_performance)
        if i == 0:
            accumulated_relative_performances.append(new_performance)
        else:
            accumulated_relative_performances.append(accumulated_relative_performances[i-1]+new_performance)
    accum_index = 0
    new_population = []
    
    index = 0
    while randoms[index] < accumulated_relative_performances[accum_index]:
        new_population.append(population[accum_index])
        accum_index += 1
        index += 1

    randoms = randoms[index:]
    for rand in randoms:
        assigned = False
        while not assigned:
            if accumulated_relative_performances[accum_index-1] < rand <= accumulated_relative_performances[accum_index]:
                new_population.append(population[accum_index])
                assigned = True
            else:
                accum_index += 1
   
    return new_population


# TODO ranking and tournament