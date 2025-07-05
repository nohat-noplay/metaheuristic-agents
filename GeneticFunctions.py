# GeneticFunctions.py 
# Author: Saf Flatters
# Year: 2025


# Functions for Run.py for Genetic Algorithm

import random as random
import numpy as np              # random for seed
import matplotlib.pyplot as plt

from SyntheticData import *

# To maintain the format provided in the Assignment Brief for global dictionaries these 2 functions will associate "id" with integers
# Use these functions instead of: 'E' + str(genome_employee), 'T' + str(genome_task)

def formatEmpID(emp_num):
    return f"E{emp_num}"

def formatTaskID(task_num):
    return f"T{task_num}"

'''
   GA Flow Chart:
1. Initial Population > 
2. Calculate Fitness Level > 
3. Selection > 
4. Crossover > 
5. Mutation > 
6. Replace >    # in GA function
7. (Repeat until Stopping Criteria <- number of generations>)
'''

### 1. Initial Population
    # 2.0 means random choice of employee per task - with sample replacement

def createChromosome():        # a list of 10 employees where task is the Index <- this is made randomly 
    chromosome = []

    employee_pool = []      # sample with replacement list of employees
    for employee in employees:
        employee_pool.append(int(employee["id"][1:]))

    for task in tasks:
        picked_employee = random.choice(employee_pool)
        chromosome.append(picked_employee)

    return chromosome


def buildPopulation(size):     # make lots of chromosomes (population)
    population = []

    for i in range(size):
        population.append(createChromosome())

    return population


### 2. Calculate Fitness Function
    # 2.0 but without unique_assignment as its not possible (NEED CLARIFICATION)

'''
Objective Function and Fitness Calculation:
    - The objective function computes the fitness score for a chromosome based on the following penalties:
        1. Overload Penalty: If an employee's total assigned task time exceeds their available hours, 
                        penalty: accumulatedtask_time - employee["hours"].
        2. Skill Mismatch Penalty: If an employee does not have the required skill for a task, 
                        penalty: +1 for every task skill not in employee skills
        3. Difficulty Violation: If an employee's skill level is lower than the task's difficulty, 
                        penalty: task["difficulty"] - employee["skill_level"]
        4. Deadline Violation: If a task's deadline is not met, a penalty is applied 
                        penalty: (currently set to 0, pending implementation).
        5. Unique Assignment Violation: This is not implemented, as it would require handling undefined task counts.
                        penalty: NONE

    - The fitness function calculates these penalties and combines them into a total score, where lower scores are better.
    - The final fitness score is the weighted sum of the individual penalties, each contributing 20% to the total score.
'''

def objectiveFunction(overload_penalty, mismatch_penalty, difficulty_violation, deadline_violation, uniqueassign_violation):      # lower the objective function the better the chromosome

    return 0.2 * overload_penalty + 0.2 * mismatch_penalty + 0.2 * difficulty_violation + 0.2 * deadline_violation  + 0.2 * uniqueassign_violation    


def matchGenomeToDict(genome_employee, genome_task):  # used by all penalty/violation functions to keep modular
    # Find the matching employee
    matched_employee = None
    for employee in employees:
        if employee["id"] == genome_employee:
            matched_employee = employee

    # Find the matching task
    matched_task = None
    for task in tasks:
        if task["id"] == genome_task:
            matched_task = task

    # Exception Handling
    if matched_employee is None:
        raise ValueError(f"Employee with ID {genome_employee} not found.")
    if matched_task is None:
        raise ValueError(f"Task with ID {genome_task} not found.")

    return matched_employee, matched_task


def calcOverload_Emp(employee, accumulatedtask_time): #overload_penalty = if task times (can be more than 1 task in 2.0 brief) is more than employee hours, return times - hours (if negative)
    if employee["hours"] < accumulatedtask_time:

        return accumulatedtask_time - employee["hours"]    
    else:
        return 0



def calcMismatch_Emp(genome_employee, genome_task):     # mismatch_penalty = if task skill is not in employee skills need to + 1
    matched_employee, matched_task = matchGenomeToDict(genome_employee, genome_task)  # to return entire dictionary line
    if matched_task["skill"] not in matched_employee["skills"]:

        return 1
    else:
        return 0


def calcDifficulty_Emp(genome_employee, genome_task):       # difficulty_violation = if employee skill_level is less than task difficulty: return skill_level - difficulty
    matched_employee, matched_task = matchGenomeToDict(genome_employee, genome_task)  # to return entire dictionary line
    if matched_employee["skill_level"] < matched_task["difficulty"]:

        return matched_task["difficulty"] - matched_employee["skill_level"]
    else: 
        return 0


def calcDeadline_Task(employee, chromosome):
    task_assignments = []

    # Collect tasks assigned to each employee
    for genome_task, genome_employee in enumerate(chromosome, start=1):
        if genome_employee == int(employee["id"][1:]):  # Compare index to ID
            _, matched_task = matchGenomeToDict(formatEmpID(genome_employee), formatTaskID(genome_task))
            task_assignments.append(matched_task)

    # Sort those tasks by processing time (Shortest Job First)
    task_assignments.sort(key=lambda task: task["time"])

    # Compute cumulative finish time and check for deadline violations
    cumulative_finish_time = 0
    total_deadline_violation = 0

    for task in task_assignments:
        cumulative_finish_time += task["time"]
        violation = max(0, cumulative_finish_time - task["deadline"])
        total_deadline_violation += violation

    return total_deadline_violation


# Unique Assignment Violation: Ensure no employee is assigned more than one task
def calcUniqueAssign_Violation(chromosome):
    task_count = {}  # Dictionary to count how many times a task is assigned
    for genome_task, genome_employee in enumerate(chromosome, start=1):
        task_id = formatTaskID(genome_task)
        if task_id not in task_count:
            task_count[task_id] = 0
        task_count[task_id] += 1

    # If any task is assigned more than once, it's a violation
    return sum(1 for count in task_count.values() if count > 1)


def fitnessCost(chromosome):  # the lower fitness level is best 
    # Employee related penalties are overload_penalty, mismatch_penalty, difficulty_violation,deadline_violation

    # create task dictionary and input each employee ID with value 0
    task_counter = {}       # eg. {'E1': [0, 0, 0, 0], 'E2': [0, 0, 0, 0], 'E3': [3, 2, 1, 1], 'E4': [15, 3, 0, 28], 'E5': [0, 0, 0, 0]}

    for employee in employees:
        task_counter[employee["id"]] = [0, 0, 0, 0]     # [overload, mismatch, difficulty, deadline]
    
    # OVERLOAD PENALTY <- need to add all hours that match the employee number (could be multiple genomes in 1 chromosome (2.0))    
    for employee in employees: 
        accumalated_task_time = 0 # there could be multiple tasks assigned to 1 employee in 2.0 so need to get accumalated amounts

        for genome_task, genome_employee in enumerate(chromosome, start=1):
            if formatEmpID(genome_employee) == employee["id"]:
                _, matched_task = matchGenomeToDict(formatEmpID(genome_employee), formatTaskID(genome_task))
                accumalated_task_time += matched_task['time']

        overload =  calcOverload_Emp(employee, accumalated_task_time)
        task_counter[employee["id"]][0] = overload

    # MISMATCH PENALTY
    for genome_task, genome_employee in enumerate(chromosome, start=1):
        mismatch = calcMismatch_Emp(formatEmpID(genome_employee), formatTaskID(genome_task))
        task_counter[formatEmpID(genome_employee)][1] += mismatch

    # DIFFICULTY VIOLATION
    for genome_task, genome_employee in enumerate(chromosome, start=1):
        difficulty = calcDifficulty_Emp(formatEmpID(genome_employee), formatTaskID(genome_task))
        task_counter[formatEmpID(genome_employee)][2] += difficulty

    # DEADLINE VIOLATION
    total_deadline_violation = sum(calcDeadline_Task(employee, chromosome) for employee in employees)

    # UNIQUE ASSIGNMENT VIOLATION
    unique_assign_violation = calcUniqueAssign_Violation(chromosome)

    # Calculate cost for entire chromosome    
    total_overload = 0
    total_mismatch = 0
    total_difficulty = 0
    total_deadline = 0

    for employee_line in task_counter:
        total_overload += task_counter[employee_line][0]
        total_mismatch += task_counter[employee_line][1]
        total_difficulty += task_counter[employee_line][2]
        total_deadline += task_counter[employee_line][3]

    return objectiveFunction(total_overload, total_mismatch, total_difficulty, total_deadline, unique_assign_violation), task_counter

### 3. Selection

'''
Elite-ism: Keep the top 20% of chromosomes for the new generation, and breed them as well.
Remove the bottom 20% of chromosomes from the breeding pool (they won't contribute to the next generation).

"Roulette Wheel Selection (RWS)": Select chromosomes based on their fitness score.
Chromosomes with lower fitness scores have a higher chance of being selected for reproduction.
'''

def rouletteSelection(ranked_chromosomes):

    inverted_fitness = []       # invert scores so higher is better
    for fitness_score, _ in ranked_chromosomes:     # for all fitness scores in the sorted chromosomes
        if fitness_score == 0:
            inverted = 1e6  # has issues with 0s
        else:
            inverted = 1 / fitness_score
        inverted_fitness.append(inverted)       # now higher is better for each chromo

    population_fitness = sum(inverted_fitness)       # get sum of all chromosome inverted fitness scores in population  

    spin_probability = []       # make wedge sizes on roulette wheel for each chromosome to sit in
    for y in inverted_fitness:
        probability = y / population_fitness
        spin_probability.append(probability)

    spin = random.random()      # spin wheel and land on a random float

    count_up_to_needle = 0      # from 0, trek through wedges to where needle has landed
    for i in range(len(spin_probability)):      
        count_up_to_needle += spin_probability[i]
        if spin <= count_up_to_needle:      # needle is found!

            winning_parent = ranked_chromosomes[i][1]

            return winning_parent      # chromosome found in that wedge - WINNING PARENT! (just chromo)


### 4. Crossover

'''
This function performs Single Point Crossover to combine the genomes of two parents (mum and dad) and create two children.
In Single Point Crossover:
    - A random crossover point is selected in the chromosome.
    - The genes (task assignments) before the crossover point are taken from the first parent (mum).
    - The genes after the crossover point are taken from the second parent (dad) for the first child, 
            and vice versa for the second child.
This process combines the genetic material of both parents to produce two offspring with mixed characteristics.
'''

def singlePointCrossover(mum, dad, chromosome_length):
    if mum is None or dad is None:
        print("ERROR: Mum or Dad is None.")
        print("Mum:", mum)
        print("Dad:", dad)
        raise ValueError("Parent selection failed.")
    crossover_point = random.randint(1, chromosome_length - 1)      # choose random no. -  subtracted 1 to avoid out of range index
    child1 = mum[ :crossover_point] + dad[crossover_point: ]
    child2 = dad[ :crossover_point] + mum[crossover_point: ]

    return child1, child2


### 5. Mutation

'''
The `reassignMutation` function randomly reassigns tasks to different employees based on a mutation rate.
For each task in the chromosome, it generates a random number and compares it with the mutation rate.
If the random number is less than the mutation rate, it selects a new employee (from the employee pool)
and assigns the task to the new employee.
This allows for exploration of new employee-task assignments, increasing diversity in the population.
'''

def reassignMutation(chromosome, mutation_rate):            # reassigns some tasks to different valid employee (using probability)

    for i in range(len(chromosome)):            # for each genome task:
        if random.random() < mutation_rate:        # probability of going further is the mutation_rate (random.random() returns a float under 1)
            genome_employee = chromosome[i]       # employee for current genome
            reassigned_employee = random.randint(1, 5)  # pick a new employee out of employee pool
            chromosome[i] = reassigned_employee  # Reassign the task to the new employee 

    return  chromosome   # put it back


### Genetic Algorithm RUN function
'''
Genetic Algorithm Implementation:
    This function runs a genetic algorithm to optimize task assignments to employees.
    It follows these steps:
    1. **Initial Population**: Generates an initial population of chromosomes (task assignments to employees).
    2. **Fitness Calculation**: Computes the fitness score for each chromosome based on objective function 
                - prints Generation 1 fitness score to user
    3. **Elite Selection**: Preserves the top 20% of the best chromosomes (elite individuals) and ensures they are part of the next generation.
                - removes bottom 20% to preserve population size
    4. **Parent Selection**: Selects two parents from the population using **roulette wheel selection**, 
                - chromosomes with lower fitness scores have a higher chance of being selected.
                - if size of population was odd, clone last one into new population
    5. **Crossover**: Performs **single-point crossover** on the selected parents to create two offspring by exchanging parts of their genomes.
    6. **Mutation**: Mutates the offspring by reassigning tasks to different employees with a 20% probability of mutating the genome.
    7. **New Population Creation**: Constructs the next generation using the elite chromosomes and the new offspring, and calculates their fitness.
    8. **Termination**: The process is repeated for a specified number of generations (specified by function call in Run.py)
    9. **Final Output**: Returns the best fitness score and the corresponding chromosome after the last generation.
'''

def geneticAlgorithm(size, generations, term=False, seed=None):

    # random seed to control randomness in plotting
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Keep track of best score per generation for plotting
    bestscore = []

    # Keep track of violations per generation for plotting
    violations_per_gen = []

    # 1. Generate Initial Population
    population = buildPopulation(size)

    # 2. Calculate initial Fitness - baseline
    ranked_population = []
    for chromosome in population:
        fitness_score, _ = fitnessCost(chromosome)  # Compute fitness (closer to 0, the better)
        ranked_population.append((fitness_score, chromosome))  # Store as a tuple

    ranked_population.sort(key=lambda x: x[0])  # sorting for printing only
    fitness_scores = []
    for fitness_score, _ in ranked_population:
        fitness_scores.append(fitness_score)
    # print("Original Population Best Objective Function Score:", fitness_scores[0])
    bestscore.append((1, fitness_scores[0]))
    last_best_score = fitness_scores[0]  
    term_count = 0

    # track violations for the best chromosome only
    best_chromosome = ranked_population[0][1]
    _, best_task_counter = fitnessCost(best_chromosome)
    violations_per_gen.append(sum(sum(v) for v in best_task_counter.values()))



# Mulitple generations
    gen_counter = 1

    for generation in range(generations):

        if gen_counter > 1:         # except for initial population so we have a baseline
           # 2. Calculate Fitness
            ranked_population = []
            for chromosome_tuple in population:
                fitness_score, _ = fitnessCost(chromosome_tuple[1])  # Compute fitness (closer to 0, the better)
                ranked_population.append((fitness_score, chromosome_tuple[1]))  # Store as a tuple

            ranked_population.sort(key=lambda x: x[0])  # sorting for printing only
            fitness_scores = []
            for fitness_score, _ in ranked_population:
                fitness_scores.append(fitness_score)
            bestscore.append((gen_counter, fitness_scores[0]))

                # track violations for the best chromosome only
            best_chromosome = ranked_population[0][1]
            _, best_task_counter = fitnessCost(best_chromosome)
            violations_per_gen.append(sum(sum(v) for v in best_task_counter.values()))

            
            # print("Population Best Objective Function Score:", fitness_scores[0])

        # ELITE-ism Preserve top 20% of chromosomes in a population (keep and also breed)
          # Split into elite and breeding pools
        elite_count = max(1, int(size * 0.2))
        elite_pool = ranked_population[:elite_count]       
        breeding_pool = ranked_population[:-elite_count]    # Everyone except the worst 20% 

        new_population = elite_pool.copy()

        while len(new_population) < size:

            # 3. Select 2 Parents
            mum = rouletteSelection(breeding_pool)        # ((fitness_score, chromosome))
            dad = rouletteSelection(breeding_pool)

            # 4. Crossover - make parent chromosomes have 2 child chromosomes
            chromosome_length = len(mum)
            child1, child2 = singlePointCrossover(mum, dad, chromosome_length)

            # 5. Mutation - change some genomes in the child chromosomes
                    # make mutated children into tuples
            
            mutated_child1_raw = reassignMutation(child1, 0.2)   # chromosome to undergo reassignment of some genomes and set mutation rate (low)
            mutated_child2_raw = reassignMutation(child2, 0.2)

            # fitness evaluation of mutated children
            mutated_child1 = (fitnessCost(mutated_child1_raw)[0], mutated_child1_raw)
            mutated_child2 = (fitnessCost(mutated_child2_raw)[0], mutated_child2_raw)   # chromosome to undergo reassignment of some genomes and set mutation rate (low)

            # 6. Add to new population
            new_population.append(mutated_child1)
            new_population.append(mutated_child2)

            if len(ranked_population) == 1:     # if there is size of population was odd, clone last one into new population
                new_population.append(ranked_population[0])

        new_population.sort(key=lambda x: x[0])
        fitness_scores = []
        for fitness_score, best_chromosome in new_population:
            fitness_scores.append(fitness_score)

        population = new_population
        gen_counter += 1

        if term:        # termination code
            if abs(population[0][0]) < 1e-6:
                break
            if term_count >= round(generations * 0.1) and term_count >= 10:  
                break   
            if last_best_score == population[0][0]:
                term_count += 1
            else:
                term_count = 0
            last_best_score = population[0][0]

    # Extract the best chromosome
    new_population.sort(key=lambda x: x[0])  # Ensure best fitness is first
    best_fitness, best_chromosome = new_population[0]  


    #print(f"Final Objective Function score: {best_fitness}")
    #print(f"Best Chromosome: {best_chromosome}")
    
    return best_chromosome, bestscore, violations_per_gen

