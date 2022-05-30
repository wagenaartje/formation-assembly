import numpy as np
from evaluation import evaluate_population, single_evaluate
from settings import *
import time

def crossover (parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    ''' Produce an offspring from 2 parents using N-point crossover '''
    offspring = np.zeros((n_param))

    choice = np.random.rand(n_param)

    offspring[choice<=0.5] = parent1[choice<=0.5]
    offspring[choice>0.5] = parent2[choice>0.5]

    return offspring

def mutate (genome: np.ndarray) -> None:
    ''' Mutates each of the genes of a genome with probability p_m '''
    for i in range(n_param):
        if np.random.rand() < p_m:
            genome[i] += np.random.rand() * 1 - 0.5

# Initialize output streams
fitness_file = open('./output/fitness.dat',mode='wb+')
genome_file = open('./output/genome.dat',mode='wb+')

# Initialize the population
population = np.random.rand(n_genomes, n_param) * 1 - 0.5

# Initialize loop variables
epoch = 1
start_time = time.time()

# Start looping
while True:
     # Crossover
    shuffle = np.random.permutation(population.shape[0])
    population = population[shuffle,:]
   
    offspring = np.zeros((n_genomes, n_param))
    for i in range(int(n_genomes / 2)):
        parent1 = population[2*i]
        parent2 = population[2*i+1]

        # With probability p_c crossover, otherwise duplicate parents
        if np.random.rand() < p_c:
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent1, parent2)
        else:
            offspring1 = parent1
            offspring2 = parent2

        offspring[2*i] = offspring1
        offspring[2*i+1] = offspring2

    # Mutation
    for i in range(n_genomes):
        mutate(offspring[i,:])

    # Evaluate the new population
    total_population = np.concatenate((population,offspring), axis=0)
    fitness = evaluate_population(total_population,n_steps)

    # Save the best genome and fitness
    best_fitness = np.min(fitness)
    best_fitness.tofile(fitness_file)

    best_genome  = total_population[[np.argmin(fitness)],:]
    best_genome.tofile(genome_file)
    
    # Tournaments selection
    new_population = np.zeros((n_genomes,n_param))

    for j in range(2): # We need two rounds!
        shuffle = np.random.permutation(total_population.shape[0])
        total_population = total_population[shuffle,:]
        fitness = fitness[shuffle]

        for i in range(int(2*n_genomes / 4)):
            fitnesses = fitness[[4*i, 4*i+1, 4*i+2, 4*i+3]]
            best_index = np.argmin(fitnesses)

            new_population[int(j*n_genomes/2 + i)] = total_population[4*i + best_index,:]

    # Update and print loop variables
    population = new_population
    print(epoch, 'Best score:', np.min(fitness), np.mean(fitnesses), time.time() - start_time, 's')
    time.sleep(0.01) # for ctrl+c
    epoch += 1
    start_time = time.time()