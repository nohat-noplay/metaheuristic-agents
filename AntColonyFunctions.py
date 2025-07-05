# AntColonyFunctions.py 
# Author: Sounness (based on Saf Flatters GeneticFunctions.py code)
# Editor: Saf Flatters
# Year: 2025


# Functions for Run.py for Ant Colony Optimisation Algorithm

import random as random
import pandas as pd
import numpy as np      # for random seed


from SyntheticData import *

# To maintain the format provided in the Assignment Brief for global dictionaries these 2 functions will associate "id" with integers
def formatEmpID(emp_num):
    return f"E{emp_num}"

def formatTaskID(task_num):
    return f"T{task_num}"


'''
ANT COLONY
1. Initialise Pheromone Matrix
2. Initialise Ants and Colony
3. Evaluate each Ant in Colony (fitness Cost)
4. Update Global Best (with best so far)
5. Evaporate Pheromones in Pheromone Matrix at an Evaporation rate
6. Deposit Pheromones - more pheromones for better (lower) fitness cost
7. Check if termination condition met
8. Loop 2 - 6 until Generations are complete
'''

#1. Initialise Pheromone Matrix

def buildPhermoneMatrix(): # initialise pheromone matrix with tasks as rows and employees as columns
    task_ids = [int(task['id'][1:]) for task in tasks]  # extract task IDs from the tasks dictionary
    employee_ids = [int(employee['id'][1:]) for employee in employees]  # extract employee IDs from the employees dictionary
    data = []
    for _ in task_ids:
        row = []
        for _ in employee_ids:
            row.append(1.0) # set all values to 1
        data.append(row)
    pheromone_matrix = pd.DataFrame(data, columns=employee_ids, index=task_ids) # create DataFrame with task IDs as index and employee IDs as columns

    return pheromone_matrix


#2. Create Ants and Colony

def createAnt(pheromone_matrix):
    ant = []
    
    for task_idk in range(len(tasks)):
        pheromones = pheromone_matrix.iloc[task_idk].values # get pheromone values for current task
        total_pheromones = sum(pheromones)  #total phermone for current task

        probabilities = []
        for pheromone in pheromones:
            probabilities.append(pheromone/total_pheromones) #probability of each employee in current task

        cummulative = []
        runnning_probability = 0
        for probability in probabilities:
            runnning_probability += probability
            cummulative.append(runnning_probability) #create cummulative probability list

        r = random.random()
        chosen_employee = None
        for emp_idk,prob in enumerate(cummulative):
            if r <= prob:   # if random less than or equal to current cummulative probability
                chosen_employee = pheromone_matrix.columns[emp_idk] 
                break

        if chosen_employee is None:
            chosen_employee = pheromone_matrix.columns[-1] #account for floating point errors

        ant.append(int(chosen_employee))
    return ant
                                                           
    
def buildColony(num_ants,pheromone_matrix):
   
    colony = []
    for i in range(num_ants):
        colony.append(createAnt(pheromone_matrix)) #create new ant for every iteration
    
    return colony


#3. Evaluate Fitness Cost of Ant in Colony
"""
##############

Fitness Cost Function from GA - converted to ACO

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
        5. Unique Assignment Violation: Ensure no employee is assigned more than one task

    - The fitness function calculates these penalties and combines them into a total score, where lower scores are better.
    - The final fitness score is the weighted sum of the individual penalties, each contributing 20% to the total score.

##############
"""

def objectiveFunction(overload_penalty, mismatch_penalty, difficulty_violation, deadline_violation, uniqueassign_violation):      # lower the objective function the better the ant

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


def calcDeadline_Task(employee, ant):
    task_assignments = []

    for genome_task, genome_employee in enumerate(ant, start=1):
        if genome_employee == int(employee["id"][1:]):  # Compare index to ID
            _, matched_task = matchGenomeToDict(formatEmpID(genome_employee), formatTaskID(genome_task))
            task_assignments.append(matched_task)

    task_assignments.sort(key=lambda x: x["time"])

    cumulative_finish_time = 0
    total_deadline_violation = 0

    for task in task_assignments:
        cumulative_finish_time += task["time"]
        violation = max(0, cumulative_finish_time - task["deadline"])
        total_deadline_violation += violation

    return total_deadline_violation


# Unique Assignment Violation: Ensure no employee is assigned more than one task
def calcUniqueAssign_Violation(ant):
    task_count = {}  # Dictionary to count how many times a task is assigned
    for genome_task, genome_employee in enumerate(ant, start=1):
        task_id = formatTaskID(genome_task)
        if task_id not in task_count:
            task_count[task_id] = 0
        task_count[task_id] += 1

    # If any task is assigned more than once, it's a violation
    return sum(1 for count in task_count.values() if count > 1)


def fitnessCost(ant):  # the lower fitness level is best 
    # Employee related penalties are overload_penalty, mismatch_penalty, difficulty_violation,deadline_violation

    # create task dictionary and input each employee ID with value 0
    task_counter = {}       # eg. {'E1': [0, 0, 0, 0], 'E2': [0, 0, 0, 0], 'E3': [3, 2, 1, 1], 'E4': [15, 3, 0, 28], 'E5': [0, 0, 0, 0]}

    for employee in employees:
        task_counter[employee["id"]] = [0, 0, 0, 0]     # [overload, mismatch, difficulty, deadline]
    
    # OVERLOAD PENALTY <- need to add all hours that match the employee number (could be multiple genomes in 1 ant (2.0))    
    for employee in employees: 
        accumalated_task_time = 0 # there could be multiple tasks assigned to 1 employee in 2.0 so need to get accumalated amounts

        for genome_task, genome_employee in enumerate(ant, start=1):
            if formatEmpID(genome_employee) == employee["id"]:
                _, matched_task = matchGenomeToDict(formatEmpID(genome_employee), formatTaskID(genome_task))
                accumalated_task_time += matched_task['time']

        overload =  calcOverload_Emp(employee, accumalated_task_time)
        task_counter[employee["id"]][0] = overload

    # MISMATCH PENALTY
    for genome_task, genome_employee in enumerate(ant, start=1):
        mismatch = calcMismatch_Emp(formatEmpID(genome_employee), formatTaskID(genome_task))
        task_counter[formatEmpID(genome_employee)][1] += mismatch

    # DIFFICULTY VIOLATION
    for genome_task, genome_employee in enumerate(ant, start=1):
        difficulty = calcDifficulty_Emp(formatEmpID(genome_employee), formatTaskID(genome_task))
        task_counter[formatEmpID(genome_employee)][2] += difficulty

    # DEADLINE VIOLATION
    total_deadline_violation = sum(calcDeadline_Task(employee, ant) for employee in employees)

    # UNIQUE ASSIGNMENT VIOLATION
    unique_assign_violation = calcUniqueAssign_Violation(ant)

    # Calculate cost for entire ant    
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


#5. Evaporate Pheromones
# Apply evaporation to all pheromone values (to reduce influence over time).

def evaporatePheromone(pheromone_matrix, evaporationRate = 0.2):   #Evap rate chosen 

    for row in pheromone_matrix.index:
        for col in pheromone_matrix.columns:
            pheromone_matrix.loc[row,col] *= (1-evaporationRate) #update all pheromone values by evaporation rate
    return pheromone_matrix

#6. Deposit   
# Deposit Pheromones - more pheromones for better (lower) fitness cost
def depositPheromone(pheromone_matrix, colony, ant_scores, Q=0.04):  #Q chosen
    sorted_colony = colony
    sorted_scores = ant_scores
   
    for i in range(len(sorted_scores)): #iterate through eat ant in colony
        min_index = i
        for j in range(i+1, len(sorted_scores)):    #sort ants in ascending order of fitness cost
            if sorted_scores[j] < sorted_scores[min_index]:
                min_index = j
        sorted_scores[i], sorted_scores[min_index] = sorted_scores[min_index], sorted_scores[i]
        sorted_colony[i], sorted_colony[min_index] = sorted_colony[min_index], sorted_colony[i]
 
    num_elite = round(len(sorted_colony) * 0.25)    # find number of elite ants (25% of colony size)
    if num_elite <= 0:
        num_elite = 1

    for i in range(num_elite):  #deposit pheromones of elite ants
        ant = sorted_colony[i]
        cost = sorted_scores[i]


        for i in range(len(ant)):
            task_idx = i+1
            emp_idx = ant[i]
            pheromone_matrix.loc[task_idx, emp_idx] = pheromone_matrix.loc[task_idx, emp_idx] + (Q / (cost + 1e-6)) #update pheromone and account for 0 cost

    return pheromone_matrix


# Function Main code

def antColonyOptimisation(num_ants, generations, term=False, seed=None):

    # random seed to control randomness in plotting
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
   
    # store gloabl bests
    global_best_ant = None
    global_best_score = float('inf')
    last_best_score = float('inf')
    term_count = 0

    # Keep track of violations per generation - global best
    violations_gbest = []
  
  #1. Initialise pheromone matrix
    pheromone_matrix = buildPhermoneMatrix()

    generation_scores = []      # need this for run.py and plotting to store all generations
   
    for generation in range(generations):

    
        #2. Initialise ants and colony
        ant_scores = []
        colony = buildColony(num_ants, pheromone_matrix) #build new ants for every iteration
        
        #3. Evaluate each ant in colony
        for ant in colony:
            cost, task_counter = fitnessCost(ant)
            ant_scores.append(cost) #store fitness cost of each ant

            #4. Update global bests
            if cost < global_best_score:   # update global bests
                global_best_score = cost
                global_best_ant = ant
                global_best_violations = sum(sum(v) for v in task_counter.values())


       
        #5. Evaporate pheromones
        pheromone_matrix = evaporatePheromone(pheromone_matrix)     # ability to alter evaporation rate in function

        #6. Deposit
        pheromone_matrix = depositPheromone(pheromone_matrix,colony,ant_scores)     # ability to alter Q in function
        
        generation_scores.append(global_best_score)         # for plotting and run.py to store all generations
        # print(f"Colony generation: {generation + 1}, Best Global Score: {global_best_score}")  #print each generation - to be commented out after testing
        violations_gbest.append(global_best_violations)

        #7. Check for termination condition

        if term:  # check for termination
            if  abs(global_best_score) < 1e-6: #optimal solution found
                break
            if term_count >= round(generations * 0.1) and term_count >= 10:   # fitness score does not change for 10 generations and 10% of total generations
                break   
            if last_best_score == global_best_score:    # if the best score has not changed
                term_count += 1 #increase counter
            else:
                term_count = 0  # reset counter if score changes
            last_best_score = global_best_score # update last best score

            
        

    #8. Return global bests
    return global_best_ant, generation_scores, violations_gbest       # now returns all generations best scores


# Run the algorithm - comment out after testing

#best_ant, best_ant_score = antColonyOptimisation(20,100,False)
#print("Best assignment:\n",best_ant)
#("Best score\n",best_ant_score[-1])





       
