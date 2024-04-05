import json
import yaml
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os


CLASS_TO_USE_INDEX = 0

with open('graphs_config.yaml', 'r') as f:
    graph_config = yaml.load(f,  Loader=yaml.FullLoader)

def show_crossover_mutation():
    results = []
    for filename in glob.glob('results_crossover_mutation/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))

    df = pd.DataFrame(results)
    df = pd.json_normalize(results)
    
    # Best solution
    plt.figure()   
    grouped = df.groupby('config.class_name')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)

    plt.figure()

    # Prueba de concepto. Mezclar clases hace que no se entienda el gráfico y agrega ruido. Habria que elegir una clase y listo
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors
    class_names = df['results.best.solution.class_name'].unique()

    for i, name in enumerate(class_names):
        filtered = df[df['results.best.solution.class_name'] == name]
        grouped = filtered.groupby('config.population_to_keep')
        plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6, color=colors[i % len(colors)])


    plt.xlabel('Population to keep')
    plt.ylabel('Performance')
    plt.title('Population to keep vs Performance for each class')
    plt.legend(class_names)

    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    grouped = filtered.groupby('config.population_size')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)
    plt.xlabel('Population size')
    plt.ylabel('Performance')
    plt.title(f'Population size vs Performance for class {class_names[CLASS_TO_USE_INDEX]}')
    plt.legend(class_names)

    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    grouped = filtered.groupby('config.mutation_config.mutation_type')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)
    plt.xlabel('Mutation type')
    plt.ylabel('Performance')
    plt.title(f'Mutation type vs Performance for class {class_names[CLASS_TO_USE_INDEX]}')


    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    grouped = filtered.groupby('config.crossover_config.method')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)

    plt.xlabel('Crossover method')
    plt.ylabel('Performance')
    plt.title(f'Crossover method vs Performance for class {class_names[CLASS_TO_USE_INDEX]}')

    for name in df["config.crossover_config.method"].unique():
        filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        by_crossover = filtered[filtered['config.crossover_config.method'] == name]
        
        plt.figure()
        filtered = by_crossover[by_crossover['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        filtered = filtered.sort_values('config.mutation_config.mutation_rate')
        grouped = filtered.groupby('config.mutation_config.mutation_rate')
        plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)
    
        plt.xlabel('Mutation rate')
        plt.ylabel('Performance')
        plt.title(f'Mutation rate vs Performance for class {class_names[CLASS_TO_USE_INDEX]} with crossover method {name}')
        plt.legend(by_crossover["config.mutation_config.mutation_rate"].unique())

        plt.figure()
        filtered = by_crossover[by_crossover['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        filtered = filtered.sort_values('config.mutation_config.delta_mutation')
        grouped = filtered.groupby('config.mutation_config.delta_mutation')
        plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

        plt.xlabel('Delta mutation')
        plt.ylabel('Performance')
        plt.title(f'Delta mutation vs Performance for class {class_names[CLASS_TO_USE_INDEX]} with crossover method {name}')
        plt.legend(by_crossover["config.mutation_config.mutation_type"].unique())

        # Best individual by generation
        # TODO change when we change the data
        plt.figure()
        filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        for mutation_type in filtered['config.mutation_config.mutation_type'].unique():
            by_mutation = filtered[filtered['config.mutation_config.mutation_type'] == mutation_type]
            generations = {}
            for data in by_mutation['results.last_iterations']:
                for character in data:
                    performance = character['best']['performance']
                    generation = character['generation']
                    generation_old = generations[generation] if generation in generations else (0, 0)
                    generations[generation] = (generation_old[0] + 1, generation_old[1] + performance)
            
            generations = {k: v[1] / v[0] for k, v in generations.items()}
            
            plt.plot(generations.keys(), generations.values(), label=mutation_type)       
            
        plt.xlabel('Generation')
        plt.ylabel('Performance')
        plt.title(f'Performance by generation for class {class_names[CLASS_TO_USE_INDEX]}')
        plt.legend()

def show_selection_tournaments():
    # Tournament performance
    results = []
    for filename in glob.glob('results/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))

    df = pd.DataFrame(results)
    df = pd.json_normalize(results)

    class_names = df['results.best.solution.class_name'].unique()


    # Performance by tournament size
    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    # Given a threshold
    filtered = filtered.sort_values('config.selection_config_a.tournament_size')
    grouped = filtered.groupby('config.selection_config_a.tournament_size')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Tournament size')
    plt.ylabel('Performance')
    plt.title(f'Performance by tournament size for class {class_names[CLASS_TO_USE_INDEX]} with threshold 0.8')

    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    # Given a tournament size
    filtered = filtered.sort_values('config.selection_config_a.tournament_size')
    grouped = filtered.groupby('config.selection_config_a.tournament_size')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Tournament size')
    plt.ylabel('Performance')
    plt.title(f'Performance by tournament size for class {class_names[CLASS_TO_USE_INDEX]}')

    # Performance by threshold
    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    filtered = filtered.sort_values('config.selection_config_a.threshold')
    grouped = filtered.groupby('config.selection_config_a.threshold')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Threshold')
    plt.ylabel('Performance')
    plt.title(f'Performance by threshold for class {class_names[CLASS_TO_USE_INDEX]}')

def show_selection_boltzmann():
    # Boltzmann performance
    # TODO REDO. Graphing params that influence each other, no good data. 
    # Graph decay, initial, final independently from each other

    results = []
    for filename in glob.glob('results_selection_boltzmann/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))

    df = pd.DataFrame(results)
    df = pd.json_normalize(results)

    class_names = df['results.best.solution.class_name'].unique()

    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    # Use final temperature 4
    filtered = filtered[filtered['config.selection_config_a.min_temperature'] == 4]
    # Use decay 0.6
    filtered = filtered[filtered['config.selection_config_a.temperature_decay'] == 0.5]
    filtered = filtered.sort_values('config.selection_config_a.initial_temperature')
    grouped = filtered.groupby('config.selection_config_a.initial_temperature')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Initial temperature')
    plt.ylabel('Performance')
    plt.title(f'Performance by initial temperature for class {class_names[CLASS_TO_USE_INDEX]}')

    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    filtered = filtered.sort_values('config.selection_config_a.initial_temperature')
    grouped = filtered.groupby('config.selection_config_a.initial_temperature')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Initial temperature')
    plt.ylabel('Performance')
    plt.title(f'Performance by initial temperature for class {class_names[CLASS_TO_USE_INDEX]}')
    

    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    filtered = filtered.sort_values('config.selection_config_a.temperature_decay')
    grouped = filtered.groupby('config.selection_config_a.temperature_decay')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Temperature decay')
    plt.ylabel('Performance')
    plt.title(f'Performance by temperature decay for class {class_names[CLASS_TO_USE_INDEX]}')


    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    filtered = filtered.sort_values('config.selection_config_a.min_temperature')
    grouped = filtered.groupby('config.selection_config_a.min_temperature')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.xlabel('Temperature decay')
    plt.ylabel('Performance')
    plt.title(f'Performance by min temperature for class {class_names[CLASS_TO_USE_INDEX]}')


def main():
    matplotlib.use('TkAgg')

    if graph_config['show_crossover_mutation']:
        show_crossover_mutation()
    
    if graph_config['show_selection']['tournaments']:
        show_selection_tournaments()

    if graph_config['show_selection']['boltzmann']:
        show_selection_boltzmann()
   
    
    # Done figures:
    # - Mutation rate vs performance
    # - Delta mutation vs performance
    # - Performance by generation for each mutation type
    # Best individual by generation

    # TODO figures:
    # Average performance by generation PENDING
    # For crossover:
        # - Performance by generation for each crossover type
    # For selection:
        # - Performance by selections and ratio
        # - Performance by temperature of boltzmann
        # For both tournaments:
            # - Performance by tournament size
            # - Performance by threshold
         

    plt.show()

def trimmed_mean(x):
    return x[x.between(x.quantile(.1), x.quantile(.9))].mean()

def trimmed_std(x):
    return x[x.between(x.quantile(.1), x.quantile(.9))].std()


if __name__ == '__main__':
    main()