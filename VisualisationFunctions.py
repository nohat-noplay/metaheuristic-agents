# VisualisationFunctions.py 
# Author: Saf Flatters
# Year: 2025


# Plotting Functions for all 3 Algorithms - called from Run.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time as time

from GeneticFunctions import geneticAlgorithm
from ParticleSwarmFunctions import particleSwarm
from AntColonyFunctions import antColonyOptimisation


# 1. Solution Quality (Optimality) - Plot a and Plot b

'''
Plot a & b
This function runs controlled benchmarking experiments to evaluate the performance with confidence intervals (3 runs) 
across two key parameters:

1. Number of Generations / Iterations       # plot a
2. Population Size                          # plot b 

Helps identify:
- Optimal number of generations 
- Optimal population size 
- Performance comparison between Algorithms

to use: 
call plotOptimalityExperiment(mode=) for 1 or 2. 
mode='generations' for 1.
mode='size' for 2. 
'''

def plotFindOptimal(mode):  # create dataframe

    gen_list = [10, 50, 100, 150, 500]
    size_list = [5, 10, 15, 20, 30]
    runs = 3

    results = []

    if mode in ('generations'):
        for gens in gen_list:
            for r in range(runs):
                seed = hash((r, gens)) % (2**32)  # reproducible runs
                _, GAscores, _ = geneticAlgorithm(20, gens, seed=seed)
                results.append({
                    'Algorithm': 'Genetic Algorithm',
                    'Variable': gens,
                    'Metric': 'Generations',
                    'FinalCost': GAscores[-1][1]
                })
                _, PSOscores, _ = particleSwarm(20, gens, seed=seed)
                results.append({
                    'Algorithm': 'Particle Swarm Optimisation',
                    'Variable': gens,
                    'Metric': 'Generations',
                    'FinalCost': PSOscores[-1]
                })
                _, ACOscores, _ = antColonyOptimisation(20, gens, seed=seed)
                results.append({
                    'Algorithm': 'Ant Colony Optimisation',
                    'Variable': gens,
                    'Metric': 'Generations',
                    'FinalCost': ACOscores[-1]
                })

    if mode in ('size'):
        for size in size_list:
            for r in range(runs):
                seed = hash((r, size)) % (2**32)  # reproducible runs
                _, GAscores, _ = geneticAlgorithm(size, 50, seed=seed)
                results.append({
                    'Algorithm': 'Genetic Algorithm',
                    'Variable': size,
                    'Metric': 'Size',
                    'FinalCost': GAscores[-1][1]
                })
                _, PSOscores, _ = particleSwarm(size, 50, seed=seed)
                results.append({
                    'Algorithm': 'Particle Swarm Optimisation',
                    'Variable': size,
                    'Metric': 'Size',
                    'FinalCost': PSOscores[-1]
                })
                _, ACOscores, _ = antColonyOptimisation(size, 50, seed=seed)
                results.append({
                    'Algorithm': 'Ant Colony Optimisation',
                    'Variable': size,
                    'Metric': 'Size',
                    'FinalCost': ACOscores[-1]
                })

    df = pd.DataFrame(results)

    if mode == 'generations':
        plot_vs_generations(df)
    else:       # mode == 'size':
        plot_vs_population_size(df)


def plot_vs_generations(df):        # plot generations
    gen_df = df[df["Metric"] == "Generations"]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=gen_df,
        x='Variable',
        y='FinalCost',
        hue='Algorithm',
        marker='o',
        estimator='mean',
        errorbar='sd',
        palette='Set2'
    )
    plt.title("Find Optimal Generation Amount \n mulitple runs, population = 20")
    plt.xlabel("Generation/Iteration/Colony")
    plt.ylabel("Final Objective Function Value")
    plt.tight_layout()
    plt.show()


def plot_vs_population_size(df):    # plot population sizes
    size_df = df[df["Metric"] == "Size"]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=size_df,
        x='Variable',
        y='FinalCost',
        hue='Algorithm',
        marker='o',
        estimator='mean',
        errorbar='sd',
        palette='Set2'
    )
    plt.title("Find Optimal Population Size \n mulitple runs, generations = 50)")
    plt.xlabel("Population Size")
    plt.ylabel("Final Objective Function Value")
    plt.tight_layout()
    plt.show()


'''
Plot c
With fixed 100 generations and size 10 - this plot shows convergence behaviour
Convergence behaviour is where the solution quality plateaus along gnerations
'''
def plotOptimality():
    generations = 100
    size = 20
    seed = 123  # Fixed seed for reproducibility

    # Run each algorithm once
    _, GAscores, _ = geneticAlgorithm(size, generations, term=True, seed=seed)
    _, PSOscores, _ = particleSwarm(size, generations, term=True, seed=seed)
    _, ACOscores, _ = antColonyOptimisation(size, generations, term=True, seed=seed)

    # Build dataframe
    convergence_data = []

    for generation, score in GAscores:
        convergence_data.append({
            'Algorithm': 'Genetic Algorithm',
            'Generation': generation,
            'Fitness': score
        })

    for generation, score in enumerate(PSOscores, start=1):
        convergence_data.append({
            'Algorithm': 'Particle Swarm Optimisation',
            'Generation': generation,
            'Fitness': score
        })

    for generation, score in enumerate(ACOscores, start=1):
        convergence_data.append({
            'Algorithm': 'Ant Colony Optimisation',
            'Generation': generation,
            'Fitness': score
        })

    df = pd.DataFrame(convergence_data)

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Generation",
        y="Fitness",
        hue="Algorithm",
        marker='o',
        palette='Set2', 
        alpha=0.8
    )
    plt.title("Solution Quality (Optimality) \n 1 run, population = 20, generations = 100")
    plt.xlabel("Generation/Iteration/Colony")
    plt.ylabel("Best Objective Function Value")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()



# 2. Computational Efficiency
'''
This function benchmarks the runtime (in seconds) of each algorithm across varying 
iteration/generation counts/time, repeating each test multiple times to capture variability.
This is repeated with termination condition and without
'''
def plotRuntime(runtime_results, term=False):

    df = pd.DataFrame(runtime_results)
    sns.set(style="whitegrid")

    # Set2 color mapping
    set2_colors = {
        'Genetic Algorithm': sns.color_palette("Set2")[0],  # green
        'Particle Swarm Optimisation': sns.color_palette("Set2")[1],  # orange
        'Ant Colony Optimisation': sns.color_palette("Set2")[2],      # purple
    }

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Iterations",
        y="Runtime",
        hue="Algorithm",
        marker="o",
        palette=set2_colors,
        estimator="mean",
        errorbar="sd"
    )

    if term: 
        plt.title("Computational Efficiency (Runtime) WITH Termination Condition\n multiple runs, population = 20")
    else: 
        plt.title("Computational Efficiency (Runtime) WITHOUT Termination Condition\n multiple runs, population = 20")
    plt.xlabel("Generation/Iteration/Colony")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()



def plotComputationalEfficiency(runs=3, term=False):
    runtime_results = []
    size = 20

    for iterations in [10, 50, 100, 200, 500]:
        for r in range(runs):
            start = time.time()
            if term: 
                geneticAlgorithm(size, iterations, term=True)
            else: 
                geneticAlgorithm(size, iterations, term=False)
            runtime_results.append({
                'Algorithm': 'Genetic Algorithm',
                'Iterations': iterations,
                'Runtime': time.time() - start
            })

            start = time.time()
            if term: 
                particleSwarm(size, iterations, term=True)
            else: 
                particleSwarm(size, iterations, term=False)
            
            runtime_results.append({
                'Algorithm': 'Particle Swarm Optimisation',
                'Iterations': iterations,
                'Runtime': time.time() - start
            })

            start = time.time()
            if term: 
                antColonyOptimisation(size, iterations, term=True)
            else: 
                antColonyOptimisation(size, iterations, term=False)
            runtime_results.append({
                'Algorithm': 'Ant Colony Optimisation',
                'Iterations': iterations,
                'Runtime': time.time() - start
            })

    plotRuntime(runtime_results, term=term)



# 3. Constraint Satisfaction (feasibility)
'''
This will plot the sum of violations per generation - plot d
'''
def plotFeasibility():

    seed = 1234
    _, _, GA_violations = geneticAlgorithm(20, 100, seed=seed) 
    _, _, PSO_violations = particleSwarm(20, 100, seed=seed) 
    _, _, ACO_violations = antColonyOptimisation(20, 100, seed=seed)


    df = pd.DataFrame({
        'Generation': list(range(1, 101)),
        'Genetic Algorithm': GA_violations,
        'Particle Swarm Optimisation': PSO_violations,
        'Ant Colony Optimisation': ACO_violations
    })

    set2_colors = {
        'Genetic Algorithm': sns.color_palette("Set2")[0],  # green
        'Particle Swarm Optimisation': sns.color_palette("Set2")[1],  # orange
        'Ant Colony Optimisation': sns.color_palette("Set2")[2],      # purple
    }

    df = df.melt(id_vars='Generation', var_name='Algorithm', value_name='Constraint Violations')

    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x='Generation', 
        y='Constraint Violations', 
        hue='Algorithm',
        palette=set2_colors, 
        marker='o', 
        alpha=0.8)

    plt.title("Constraint Satisfaction (Feasibility) \n 1 run, population = 20")
    plt.xlabel("Generation/Iteration/Colony")
    plt.ylabel("Total Constraint Violations (sum of penalties)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





