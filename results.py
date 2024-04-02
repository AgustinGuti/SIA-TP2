import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

def main():
    matplotlib.use('TkAgg')
    results = []
    for filename in glob.glob('results/*.json'):
        with open(filename, 'r') as f:
          results.append(json.load(f))

    df = pd.DataFrame(results)
    df = pd.json_normalize(results)
    

    plt.figure()   
    grouped = df.groupby('config.class_name')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='o', capsize=6)

    plt.figure()
    grouped = df.groupby('config.selection_config_a.population_to_keep')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)
    plt.xlabel('Population to keep')
    plt.ylabel('Performance')
    plt.title('Population to keep vs Performance')

    plt.figure()
    grouped = df.groupby('config.population_size')
    plt.errorbar(grouped.groups.keys(), grouped['results.best.solution.performance'].mean(), yerr=grouped['results.best.solution.performance'].std(), fmt='-o', capsize=6)

    plt.show()


if __name__ == '__main__':
    main()