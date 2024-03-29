import math

def elite_selection(population, population_to_keep):
    population.sort(key=lambda x: x.performance, reverse=True)
    population_size = len(population)
    new_population = []
    for i in range(0, len(population)-1):
        n = math.ceil((population_to_keep-i)/population_size)
        new_population.extend([population[i]]*n)
    
    return new_population