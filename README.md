# Metaheuristic Agents 🧬🐝🐜
Year: 2025

This project uses a collection of nature-inspired optimisation algorithms, exploring how intelligent agent behaviours can be used to solve complex problems through swarm intelligence and evolutionary processes.

## Project Overview
This project explores the use of heuristic intelligent agents and evolutionary algorithms — specifically:
- **Genetic Algorithm** (GA) – inspired by biological evolution, using selection, crossover, and mutation.
- **Ant Colony Optimisation** (ACO) – inspired by how ants lay and follow pheromone trails to discover efficient paths.
- **Particle Swarm Optimisation** (PSO) – inspired by collective motion in bird flocks or fish schools, using shared experiences.

These algorithms are applied to an employee-task assignment problem featuring multiple real-world constraints, including:  
- Skill compatibility  
- Workload balancing  
- Task difficulty  
- Deadline adherence  
- Unique task assignments  

## Why Metaheuristics?
A metaheuristic is a high-level problem-solving strategy that guides lower-level heuristics to search large, complex, or ill-defined solution spaces efficiently — especially when:  
- The problem is computationally expensive (e.g. NP-hard),
- Multiple constraints conflict,
- Exhaustive search is impractical.  
By mimicking natural or collective behaviours, these algorithms find good-enough solutions where traditional optimisation fails.

## Features
- Modular design for running GA, PSO, and ACO independently or comparatively  
- Built-in constraint handling with penalty-based fitness evaluation  
- Visualisation tools for analysing optimisation performance  
- Synthetic task and employee data generation  

## How to Run
In the project folder terminal:  
`pip install -r requirements.txt`  

Ensure all required files are in your working directory:  
`python3 run.py`                            

## File Structure

├── `run.py`                            # Main execution script  

Supporting files:  
├── `SyntheticData.py`                  # Employee and Task data    
├── `GeneticFunctions.py`               # Genetic Algorithm implementation    
├── `ParticleSwarmFunctions.py`         # PSO implementation    
├── `AntColonyFunctions.py`             # ACO implementation    
├── `VisualisationFunctions.py`         # Visualisation utilities  

Outputs & Other:   
└── Outputs/   
--├── plots/                              # Output folder for plots      
--├── output_screenshot.png             # Sample terminal output screenshot      
└── Docs/  
--├── Report                            # Report with technical details and assessment

## Credits
Saf Flatters  
Thomas Sounness  
Nandar Ko Ko Lynn  


## Connect with Me
📫 [LinkedIn](https://www.linkedin.com/in/safflatters/)

## License and Usage
![Personal Use Only](https://img.shields.io/badge/Personal%20Use-Only-blueviolet?style=for-the-badge)

This project is intended for personal, educational, and portfolio purposes only.
You are welcome to view and learn from this work, but you may not copy, modify, or submit it as your own for academic, commercial, or credit purposes.