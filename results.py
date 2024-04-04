import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

CLASS_TO_USE_INDEX = 0

def main():
    matplotlib.use('TkAgg')
    results = []
    for filename in glob.glob('results/*.json'):
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
    plt.legend(grouped.groups.keys())


    plt.figure()
    filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
    grouped = filtered.groupby('config.crossover_config.method')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)

    plt.xlabel('Crossover method')
    plt.ylabel('Performance')
    plt.title(f'Crossover method vs Performance for class {class_names[CLASS_TO_USE_INDEX]}')
    plt.legend(grouped.groups.keys())

    for name in df["config.crossover_config.method"].unique():
        filtered = df[df['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        by_crossover = filtered[filtered['config.crossover_config.method'] == name]

        plt.figure()
        filtered = by_crossover[by_crossover['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
        grouped = filtered.groupby('config.mutation_config.mutation_type')
        plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)

        plt.xlabel('Mutation type')
        plt.ylabel('Performance')
        plt.title(f'Mutation type vs Performance for class {class_names[CLASS_TO_USE_INDEX]} with crossover method {name}')
        plt.legend(by_crossover["config.mutation_config.mutation_type"].unique())

        
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

        plt.figure()
        for i, name in enumerate(by_crossover["config.mutation_config.mutation_type"].unique()):
            filtered = by_crossover[by_crossover['results.best.solution.class_name'] == class_names[CLASS_TO_USE_INDEX]]
            filtered = filtered[filtered['config.mutation_config.mutation_type'] == name]
            filtered['mutation_avg'] = filtered['config.mutation_config.mutation_rate'] * filtered['config.mutation_config.delta_mutation']
            filtered = filtered.sort_values('mutation_avg')
            grouped = filtered.groupby('mutation_avg')
            plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

        plt.xlabel('Mutation rate * Delta mutation')
        plt.ylabel('Performance')
        plt.title(f'Mutation rate * Delta mutation vs Performance for class {class_names[CLASS_TO_USE_INDEX]} with crossover method {name}')
        plt.legend(by_crossover["config.mutation_config.mutation_type"].unique())


    # Comparacion de crossover se hace el mejor personaje para una o dos clases
        
    # 


    plt.show()


if __name__ == '__main__':
    main()