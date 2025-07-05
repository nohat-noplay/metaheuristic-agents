# ParticleSwarmFunctions.py 
# Author: Nandar-Lynn (based on Saf Flatters GeneticFunctions.py code)
# Editor: Saf Flatters
# Year: 2025

# Functions for Run.py for PSO Algorithm


import random as random
import matplotlib.pyplot as plt
import time
import numpy as np

from SyntheticData import *

# Functions to match employee and task IDs to the global dictionaries
def formatEmpID(emp_num):
    return f"E{emp_num}"

def formatTaskID(task_num):
    return f"T{task_num}"

# Initialize Swarm: Create random task-to-employee assignments (particles)
def createParticle():
    particle = []  # Each particle represents an assignment of tasks to employees

    # Get list of employee indices (from 0 to len(employees) - 1)
    employee_pool = [i for i in range(len(employees))]  # Get employee indices (0 to len(employees)-1)

    for task in tasks:
        picked_employee_index = random.choice(employee_pool)  # Randomly assign employee to task by index
        particle.append(picked_employee_index)  # Append the employee index to the particle

    return particle

def buildSwarm(size):
    return [createParticle() for _ in range(size)]  # Build a swarm with multiple particles

# Objective function to calculate fitness based on penalties
def objectiveFunction(overload_penalty, mismatch_penalty, difficulty_violation, deadline_violation, unique_assign_violation):
    return (0.2 * overload_penalty + 0.2 * mismatch_penalty + 
            0.2 * difficulty_violation + 0.2 * deadline_violation + 
            0.2 * unique_assign_violation)

# Match the employee and task by their IDs
def matchComponentToDict(component_employee, component_task):
    matched_employee = employees[component_employee]  # Access employee by index
    matched_task = next((tsk for tsk in tasks if tsk["id"] == component_task), None)

    if not matched_task:
        raise ValueError(f"Task with ID {component_task} not found.")

    return matched_employee, matched_task

# Penalty Functions
def calcOverload_Emp(employee, accumulated_task_time):
    # Overload penalty: If employee's total task time exceeds available hours, return the overflow time
    return max(0, accumulated_task_time - employee["hours"])

def calcMismatch_Emp(component_employee, component_task):
    matched_employee, matched_task = matchComponentToDict(component_employee, component_task)
    return 1 if matched_task["skill"] not in matched_employee["skills"] else 0  # Penalty for skill mismatch

def calcDifficulty_Emp(component_employee, component_task):
    matched_employee, matched_task = matchComponentToDict(component_employee, component_task)
    return max(0, matched_task["difficulty"] - matched_employee["skill_level"])  # Difficulty penalty

# Deadline Violation: Calculate penalty based on task deadlines
def calcDeadline_Task(employee, particle):
    task_assignments = []
    # Gather all tasks assigned to this employee
    for component_task, component_employee in enumerate(particle, start=1):
        if component_employee == employee["id"]:  # Use employee index
            matched_employee, matched_task = matchComponentToDict(component_employee, formatTaskID(component_task))
            task_assignments.append(matched_task)
    
    # Sort tasks assigned to the employee by processing time (ascending)
    task_assignments.sort(key=lambda x: x["time"])

    # Calculate cumulative finish times and check for deadline violations
    cumulative_finish_time = 0
    total_deadline_violation = 0

    for task in task_assignments:
        cumulative_finish_time += task["time"]  # Add the current task's processing time
        # Calculate the violation if the task exceeds its deadline
        violation = max(0, cumulative_finish_time - task["deadline"])
        total_deadline_violation += violation  # Accumulate total penalty

    return total_deadline_violation

# Unique Assignment Violation: Ensure no employee is assigned more than one task
def calcUniqueAssign_Violation(particle):
    task_count = {}  # Dictionary to count how many times a task is assigned
    for component_task, component_employee in enumerate(particle, start=1):
        task_id = formatTaskID(component_task)
        if task_id not in task_count:
            task_count[task_id] = 0
        task_count[task_id] += 1

    # If any task is assigned more than once, it's a violation
    return sum(1 for count in task_count.values() if count > 1)

# Fitness function for a particle: Calculate the total penalty for a task assignment (particle)
def fitnessCost(particle):
    task_counter = {i: [0, 0, 0, 0, 0] for i in range(len(employees))}  # Track penalties for each employee (use indices)

    # Calculate overload penalty for each employee
    for employee_index, employee in enumerate(employees):
        accumulated_task_time = sum(
            matched_task['time'] for component_task, component_employee in enumerate(particle, start=1)
            if component_employee == employee_index  # Use employee index for comparison
            for _, matched_task in [matchComponentToDict(component_employee, formatTaskID(component_task))]
        )
        overload = calcOverload_Emp(employee, accumulated_task_time)
        task_counter[employee_index][0] = overload

    # Calculate mismatch penalty (for skill mismatches)
    for component_task, component_employee in enumerate(particle, start=1):
        mismatch = calcMismatch_Emp(component_employee, formatTaskID(component_task))  # Directly pass employee index
        task_counter[component_employee][1] += mismatch

    # Calculate difficulty violation penalty
    for component_task, component_employee in enumerate(particle, start=1):
        difficulty = calcDifficulty_Emp(component_employee, formatTaskID(component_task))  # Directly pass employee index
        task_counter[component_employee][2] += difficulty

    # Calculate deadline violation (currently set to 0, can be expanded)
    total_deadline_violation = sum(calcDeadline_Task(employee, particle) for employee in employees)

    # Calculate unique assignment violation
    unique_assign_violation = calcUniqueAssign_Violation(particle)

    # Calculate total penalties
    total_overload = sum(task_counter[employee_index][0] for employee_index in task_counter)
    total_mismatch = sum(task_counter[employee_index][1] for employee_index in task_counter)
    total_difficulty = sum(task_counter[employee_index][2] for employee_index in task_counter)

    return objectiveFunction(total_overload, total_mismatch, total_difficulty, total_deadline_violation, unique_assign_violation), task_counter

# Particle Swarm Optimization (PSO) algorithm
def particleSwarm(swarm_size, iterations, term=False, seed=None):

    # random seed to control randomness in plotting
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # for feasibility plotting
    violations_gbest = 0
    violations_gbest_gen = []
    
    # Helper function to shift particle indices from 0-based to 1-based
    def shift_particle_indices(particle):
        return [x + 1 for x in particle]

    # Initialize swarm
    swarm = buildSwarm(swarm_size)
    pBest = swarm[:]  # Personal best of each particle
    pBest_fitness = [fitnessCost(p)[0] for p in swarm]  # Fitness of each particle's personal best

    # Initialize global best
    gBest = min(swarm, key=lambda p: fitnessCost(p)[0])  # Best solution found
    gBest_fitness, task_counter = fitnessCost(gBest)
    violations_gbest = sum(sum(v) for v in task_counter.values())
    last_gBest = gBest_fitness  
    term_count = 0

    # PSO loop
    best_scores = []  # To store fitness values for plotting
    for i in range(iterations):
        for j in range(swarm_size):
            # Update particle's velocity (using personal best and global best)
            particle = swarm[j]
            particle_velocity_update(particle, pBest[j], gBest)

            # Update particle's position
            particle_position_update(particle)

            # Evaluate fitness of particle
            current_fitness, task_counter = fitnessCost(particle)

            # Update personal best if current fitness is better
            if current_fitness < pBest_fitness[j]:
                pBest[j] = particle[:]
                pBest_fitness[j] = current_fitness

            # Update global best if current fitness is better
            if current_fitness < gBest_fitness:
                gBest = particle[:]
                gBest_fitness = current_fitness
                violations_gbest = sum(sum(v) for v in task_counter.values())

        # Record best score after each iteration
        best_scores.append(gBest_fitness)
        violations_gbest_gen.append(violations_gbest)
        # print(f"Iteration {i + 1}/{iterations}, Global Best Fitness: {gBest_fitness}")

        if term:
            if abs(gBest_fitness) < 1e-6:  # Early termination if fitness is zero
                break
            if term_count >= round(iterations * 0.1) and term_count >= 10:  
                break   
            if last_gBest == gBest_fitness:
                term_count += 1
            else:
                term_count = 0
            last_gBest = gBest_fitness

    return shift_particle_indices(gBest), best_scores, violations_gbest_gen

# Update particle velocity using the standard PSO formula
def particle_velocity_update(particle, pBest, gBest):
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive component
    c2 = 1.5  # Social component

    for i in range(len(particle)):
        r1 = random.random()
        r2 = random.random()
        velocity = w * (pBest[i] - particle[i]) + c1 * r1 * (pBest[i] - particle[i]) + c2 * r2 * (gBest[i] - particle[i])
        particle[i] += velocity

# Update particle's position based on the velocity
def particle_position_update(particle):
    for i in range(len(particle)):
        particle[i] = int(particle[i])  # Convert to an integer employee ID (or index)
        # Ensure the particle's position is valid (between 0 and number of employees - 1)
        particle[i] = max(0, min(particle[i], len(employees) - 1))

# # Test the PSO with example parameters
# def test_PSO():
#     swarm_size = 1  # Example swarm size
#     iterations = 2  # Number of iterations

#     best_assignment, fitness_scores = particleSwarm(swarm_size, iterations)

#     print("Best Assignment:", best_assignment)
#     print("Fitness Scores:", fitness_scores)

# # Run the test function
# test_PSO()
