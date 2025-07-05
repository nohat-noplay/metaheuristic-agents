# Run.py  
# Author: Saf Flatters
# Year: 2025


# To run all Function files and output solutions and plots

from GeneticFunctions import geneticAlgorithm
from ParticleSwarmFunctions import particleSwarm
from AntColonyFunctions import antColonyOptimisation
from SyntheticData import tasks, employees
from VisualisationFunctions import *


# Function to format results 
def humanReadable(result):
    print("+---------+-----------+")
    print("| Task #  | Employee #|")
    print("+---------+-----------+")
    for task, employee in enumerate(result, start=1):
        print(f"|  {str(task).center(6)} |  {str(employee).center(8)} |")
    print("+---------+-----------+")



### GENETIC ALGORITHM 

GAsize = 10
generations = 500


GAsolution, GAbestscores, GAviolations = geneticAlgorithm(GAsize, generations, term=True)       # (size, generations) where size = how many chromosones, generation = how many loops

print(f"~ Genetic Algorithm ~")
print(f"\nGeneration 1 Cost Score: {GAbestscores[0][1]}")
print(f"Final Generation Cost Score: {GAbestscores[-1][1]}")
print(f"Final Best Chromosome: {GAsolution}")
humanReadable(GAsolution)
print("\n")


### PARTICLE SWARM OPTIMISATION ALGORITHM
PSOsize = 10
iterations = 500
PSOsolution, PSObestscores, PSOviolations = particleSwarm(PSOsize, iterations, term=True)

print(f"~ Particle Swarm Optimisation ~")
print(f"\nIteration 1 Cost Score: {PSObestscores[0]}")
print("Final Iteration Score:", PSObestscores[-1])
print("Final Best Particle Position:", PSOsolution)
humanReadable(PSOsolution)
print("\n")

### ANT COLONY OPTIMISATION ALGORITHM

num_ants = 10
colony = 500
ACsolution, ACbestscores, ACviolations = antColonyOptimisation(num_ants, colony, term=True)       # (size, generations) where size = how many particles, iterations = how many loops
print(f"~ Ant Colony Optimisation ~")
print(f"\nColony 1 Cost Score: {ACbestscores[0]}")
print(f"Final Colony Cost Score: {ACbestscores[-1]}")
print(f"Final Best Ant: {ACsolution}")
humanReadable(ACsolution)
print("\n")



### OUTPUT PLOTS

'''
1. SOLUTION  QUALITY
(Plot a)Test multiple generations, multiple replications to produce a plot to determine:
- Final Objective Values (with confidence intervals) for each algorithm

(Plot b)Test multiple sizes, multiple replications to produce a plot to determine:
- Final Objective Values (with confidence intervals) for each algorithm

(Plot c) Test fixed 100 generations, 1 replication to produce a plot to determine
- Convergence Behaviour for each algorithm
'''

# plotFindOptimal(mode='generations')      # plot a
# plotFindOptimal(mode='size')             # plot b
# plotOptimality()                       # plot c

'''
2. COMPUTATIONAL EFFICIENCY
(Plot d & d2) Test multiple generations, multiple replications to produce a plot to determine:
- runtime performance (with confidence intervals) of each algorithm 
- the effectiveness of the Termination Condition
'''

# plotComputationalEfficiency(term=True)       # plot d
# plotComputationalEfficiency(term=False)        # plot d2


'''
3. CONTRAINT SATISFACTION (FEASIBILITY)
(Plot e) Test fixed 100 generations, 1 replication to produce a plot to determine:
- how quickly and effectively each algorithm produces solutions that satisfy constraints
'''

# plotFeasibility()       # plot e