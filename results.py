import json
import yaml
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import seaborn as sns
import seaborn.matrix as smatrix
from scipy.optimize import curve_fit

CLASS_TO_USE_INDEX = 0

with open('graphs_config.yaml', 'r') as f:
    graph_config = yaml.load(f,  Loader=yaml.FullLoader)

def calculate_generations_data(data_iterations):
    generations = {}
    # gen -> (count, sum)
    for file in data_iterations['results.last_iterations']:
        for iteration in file:
            best_performance = iteration['best']['performance']
            average_performance = iteration['mean']
            generation = iteration['generation']
            diversity = iteration['diversity']
            generation_old = generations[generation] if generation in generations else (0, 0, 0, 0)
            generations[generation] = (generation_old[0] + 1, generation_old[1] + best_performance, generation_old[2] + average_performance, generation_old[3] + diversity)

    # Average performance by generation
    average_performances = {k: v[2] / v[0] for k, v in generations.items()}
    best_performances = {k: v[1] / v[0] for k, v in generations.items()}
    diversity = {k: v[3] / v[0] for k, v in generations.items()}
    return best_performances, average_performances, diversity

def _show_performance_by_attribute_from_data(df, config, attr, attr_name, hide_individual_graphs=True, only_diversity=False, extra_title=""):
    if extra_title != "":
        extra_title = f' - {extra_title}'
   
    class_names = df['results.best.solution.class_name'].unique()

    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    filtered = filtered.sort_values(f'{config}.{attr}')
    grouped = filtered.groupby(f'{config}.{attr}')

    if not only_diversity:
        plt.figure()
        plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)
        plt.xlabel(attr_name.capitalize())
        plt.ylabel('Performance')
        plt.title(f'{attr_name.capitalize()} vs Performance for class {class_names[CLASS_TO_USE_INDEX]}{extra_title}')

        plt.figure()
        plt.errorbar(grouped.groups.keys(), grouped['results.best.generation'].mean(), yerr=grouped['results.best.generation'].std(), fmt='o', capsize=6)
        plt.xlabel(attr_name.capitalize())
        plt.ylabel('Generations')
        plt.title(f'Generations to reach best by {attr_name} for class {class_names[CLASS_TO_USE_INDEX]}{extra_title}')

    data_by_attr = {}
    for attr_type in filtered[f'{config}.{attr}'].unique():
        by_attr = filtered[filtered[f'{config}.{attr}'] == attr_type]
        data_by_attr[attr_type] = calculate_generations_data(by_attr)

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red', 'tab:green']
    colors_by_attr = {attr_type: colors.pop(0) for attr_type in data_by_attr.keys()}

    if not only_diversity:
        if not hide_individual_graphs:
            for attr_type, (best_performances, average_performances, diversity) in data_by_attr.items():
                plt.figure()
                plt.plot(best_performances.keys(), best_performances.values())   
                plt.plot(average_performances.keys(), average_performances.values())     
                plt.xlabel('Generation')
                plt.ylabel('Performance')
                plt.title(f'Performance by generation for class {class_names[CLASS_TO_USE_INDEX]} with {attr_type} {attr_name}{extra_title}')
                plt.legend(['Best', 'Average'])

        plt.figure()
        for attr_type, (best_performances, average_performances, diversity) in data_by_attr.items():
            plt.plot(best_performances.keys(), best_performances.values(), colors_by_attr[attr_type])
        plt.xlabel('Generation')
        plt.ylabel('Performance')
        plt.title(f'Performance by generation for class {class_names[CLASS_TO_USE_INDEX]}{extra_title}')
        legend = plt.legend(filtered[f'{config}.{attr}'].unique())
        legend.set_title(f'{attr_name.capitalize()}')

        plt.figure()
        for attr_type, (best_performances, average_performances, diversity) in data_by_attr.items():
            plt.plot(average_performances.keys(), average_performances.values(), colors_by_attr[attr_type])
        plt.xlabel('Generation')
        plt.ylabel('Performance')
        plt.title(f'Average performance by generation for class {class_names[CLASS_TO_USE_INDEX]}{extra_title}')
        legend = plt.legend(filtered[f'{config}.{attr}'].unique())
        legend.set_title(f'{attr_name.capitalize()}')

    # Plot diversity
    plt.figure()
    for attr_type, (best_performances, average_performances, diversity) in data_by_attr.items():       
        plt.plot(diversity.keys(), diversity.values(), colors_by_attr[attr_type])

    plt.xlabel('Generation')
    plt.ylabel('Diversity')
   
    plt.title(f'Diversity by generation for class {class_names[CLASS_TO_USE_INDEX]}{extra_title}')
    legends = [f'{x}' for i, x in enumerate(filtered[f'{config}.{attr}'].unique())]
    legend = plt.legend(legends)
    legend.set_title(f'{attr_name.capitalize()}')

def show_performance_by_attribute(config, attr, attr_name, folder_name=None, hide_individual_graphs=True):
    if folder_name is None:
        folder_name = attr
    results = []
    for filename in glob.glob(f'results/performance_by_gen/{folder_name}/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))
        
    df = pd.DataFrame(results)
    df = pd.json_normalize(results)

    _show_performance_by_attribute_from_data(df, config, attr, attr_name, hide_individual_graphs)
   
def show_selection_tournaments():
    # Tournament performance
    results = []
    for filename in glob.glob('results/results_selection_tournaments/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))

    df = pd.DataFrame(results)
    df = pd.json_normalize(results)

    class_names = df['results.best.solution.class_name'].unique()


    # Performance by tournament size
    plt.figure()
    # Given a threshold
    df = df.sort_values('config.selection_config_a.threshold')
    for threshold in df['config.selection_config_a.threshold'].unique():
        filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        filtered = filtered[filtered['config.selection_config_a.threshold']==threshold]
        filtered = filtered.sort_values('config.selection_config_a.tournament_size')
        grouped = filtered.groupby('config.selection_config_a.tournament_size')
        means = grouped['results.best.generation'].agg(trimmed_mean)
        errors = grouped['results.best.generation'].agg(trimmed_std)
        plt.plot(grouped.groups.keys(), grouped['results.best.generation'].mean())
        # plt.errorbar(grouped.groups.keys(), grouped['results.best.generation'].mean(), yerr=grouped['results.best.generation'].std(), fmt='-o', capsize=6)

    
    plt.xlabel('Tournament size')
    plt.ylabel('Performance')
    plt.title(f'Performance by tournament size for class {class_names[CLASS_TO_USE_INDEX]} for each threshold')
    plt.legend( df['config.selection_config_a.threshold'].unique())

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
    for filename in glob.glob('results/results_selection_boltzmann/*.json'):
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
    # 3D plot - initial, decay vs performance
    # Use final temperature 4
    filtered = filtered[filtered['config.selection_config_a.min_temperature'] == 4]
    X = filtered['config.selection_config_a.initial_temperature'].to_numpy()
    Y = filtered['config.selection_config_a.temperature_decay'].to_numpy()
    grouped = filtered.groupby(['config.selection_config_a.initial_temperature', 'config.selection_config_a.temperature_decay'])
    Z = grouped['results.best.solution.performance'].mean().to_numpy()

    X_unique = np.sort(np.unique(X))
    Y_unique = np.sort(np.unique(Y))
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
    Z_grid = Z.reshape(X_grid.shape)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z_grid)
    ax.set_xlabel('Initial temperature')
    ax.set_ylabel('Temperature decay')
    ax.set_zlabel('Performance')
    plt.title(f'Performance by initial temperature and decay for class {class_names[CLASS_TO_USE_INDEX]}')




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

def show_selection_combinations():
    results = []
    for filename in glob.glob('results/performance_by_gen/selection_ratios/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))
        
    df = pd.DataFrame(results)
    df = pd.json_normalize(results)

    class_names = df['results.best.solution.class_name'].unique()
    df = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]

    df['config.selection_ratio'] = df['config.selection_config_a.population_to_keep'] / df['config.population_to_keep']
    
    # I want to create a 21x5 matrix with the performance of each combination
    # I will use a heatmap to show the data

    # Create a matrix with the performance of each combination
    data_map = {}
    already_done_pairs = []
    df = df.sort_values('config.selection_ratio')

    for i, selection_type_a in enumerate(df['config.selection_config_a.selection_type'].unique()):
        for j, selection_type_b in enumerate(df['config.selection_config_b.selection_type'].unique()):
            if selection_type_a == selection_type_b or (selection_type_b, selection_type_a) in already_done_pairs:
                continue
            has_data = False
            data_map[(selection_type_a, selection_type_b)] = np.zeros((len(df['config.selection_ratio'].unique())))

            for k, ratio in enumerate(df['config.selection_ratio'].unique()):
                filtered = df[(df['config.selection_config_a.selection_type'] == selection_type_a) & (df['config.selection_config_b.selection_type'] == selection_type_b) & (df['config.selection_ratio'] == ratio)]
                if not np.isnan(filtered["results.best.solution.performance"].mean()):
                    has_data = True
                

                data_map[(selection_type_a, selection_type_b)][k] = filtered['results.best.solution.performance'].mean()
                # matrix[len(already_done_pairs) - 1, k] = filtered['results.best.solution.performance'].mean()

            if has_data:
                already_done_pairs.append((selection_type_a, selection_type_b))
            else:
                data_map.pop((selection_type_a, selection_type_b))
                
    matrix = np.zeros((len(data_map), len(df['config.selection_ratio'].unique())))
    for i, (key, value) in enumerate(data_map.items()):
        matrix[i] = value

    # Create heatmap using seaborn
    plt.figure()
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap='YlGn', yticklabels=data_map.keys(), xticklabels=df['config.selection_ratio'].unique())
    plt.xlabel('Ratio')
    plt.ylabel('Selection combination')
    plt.title(f'Performance by selection combination for class {class_names[CLASS_TO_USE_INDEX]}')

    # Now I want to create a graph for each combination, with the performance by generation
    
    for i, (key, value) in enumerate(data_map.items()):
        partial = df[(df['config.selection_config_a.selection_type'] == key[0]) & (df['config.selection_config_b.selection_type'] == key[1])]
        _show_performance_by_attribute_from_data(partial, 'config', 'selection_ratio', 'ratio', only_diversity=True, extra_title=f'{key[0]} and {key[1]}')


def main():
    matplotlib.use('TkAgg')

    if graph_config['show_mutation']['by_type']:
        show_performance_by_attribute('config.mutation_config', 'mutation_type', 'mutation type')

    if graph_config['show_mutation']['by_rate']:
        show_performance_by_attribute('config.mutation_config', 'mutation_rate', 'mutation rate')

    if graph_config['show_mutation']['by_delta']:
        show_performance_by_attribute('config.mutation_config', 'delta_mutation', 'delta mutation', "mutation_delta")
   
    if graph_config['show_crossover_performance']:
        show_performance_by_attribute('config.crossover_config', 'method', 'crossover method', 'crossover')

    if graph_config['show_algorithm']['by_population_size']:
        show_performance_by_attribute('config', 'population_size', 'population size')

    if graph_config['show_algorithm']['by_population_to_keep']:
        show_performance_by_attribute('config', 'population_to_keep', 'population to keep', 'population_to_keep')

    if graph_config['show_replacement']:
        show_performance_by_attribute('config.replacement_config_a', 'gen_gap', 'generational gap', 'replacement_config')

    # TODO check implementation and result analysis. Something is off
    if graph_config['show_selection']['type']:
        show_performance_by_attribute('config.selection_config_a', 'selection_type', 'selection type')

    if graph_config['show_selection']['boltzmann_initial_temp']:
        show_performance_by_attribute('config.selection_config_a', 'initial_temperature', 'initial temperature', 'boltzmann_initial_temp')

    if graph_config['show_selection']['boltzmann_decay']:
        show_performance_by_attribute('config.selection_config_a', 'temperature_decay', 'temperature decay', 'boltzmann_decay')

    if graph_config['show_selection']['boltzmann_min_temp']:
        show_performance_by_attribute('config.selection_config_a', 'min_temperature', 'min temperature', 'boltzmann_min_temp')

    if graph_config['show_selection']['tournament_size']:
        show_performance_by_attribute('config.selection_config_a', 'tournament_size', 'tournament size', 'tournament_size')


    if graph_config['show_selection']['combination']:
        show_selection_combinations()


    # TODO migrate to show_performance_by_attribute
    # if graph_config['show_selection']['tournaments']:
    #     show_selection_tournaments()

    # if graph_config['show_selection']['boltzmann']:
    #     show_selection_boltzmann()

    # Done figures:
    # - Mutation rate vs performance CHECK
    # - Delta mutation vs performance CHECK
    # - Performance by generation for each mutation type CHECK
    # Best individual by generation CHECK
    # For crossover:
        # - Performance by generation for each crossover type DONE
        # - Performance by crossover type DONE
        # - Generations to reach best by crossover type DONE
        

    # TODO figures:
    # Average performance by generation ADD WITH OTHER CONFIGS
   
    # For selection:
        # - Performance by selections and ratio
        # - Performance by temperature of boltzmann
        # For both tournaments:
            # - Performance by tournament size
            # - Performance by threshold
         

    plt.show()

def trimmed_mean(x):
    return x[x.between(x.quantile(.2), x.quantile(.8))].mean()

def trimmed_std(x):
    return x[x.between(x.quantile(.2), x.quantile(.8))].std()


if __name__ == '__main__':
    main()