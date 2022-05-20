from cv2 import absdiff
import numpy as np
from evaluation import evaluate_population, single_evaluate
from settings import *
import time

def crossover (parent1, parent2):
    offspring = np.zeros((n_param))

    choice = np.random.rand(n_param)

    offspring[choice<=0.5] = parent1[choice<=0.5]
    offspring[choice>0.5] = parent2[choice>0.5]

    return offspring

# NOTE! this function is quite slow
def mutate (genome):
    
    for i in range(n_param):
        if np.random.rand() < p_m:
            genome[i] += np.random.rand() * 2 -1

population = np.random.rand(n_genomes, n_param) * 2 - 1

best_genome = None
best_fitness = np.inf
best_epoch = None
epoch = 0
start_time = time.time()
while True:
    ''' Stage I: Create n offspring '''
     # Crossover
    shuffle = np.random.permutation(population.shape[0])
    population = population[shuffle,:]
   
    offspring = np.zeros((n_genomes, n_param))
    for i in range(int(n_genomes / 2)):
        parent1 = population[2*i]
        parent2 = population[2*i+1]

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

    ''''Stage II: Tournament selection '''
    total_population = np.concatenate((population,offspring), axis=0)
    fitness = evaluate_population(total_population)

    # NOTE! There is randomness, so genome will behave differently.
    best_genome = total_population[[np.argmin(fitness)],:]
    single_evaluate(best_genome,save=True)

    #if epoch - best_epoch > 5:
    #    break

    new_population = np.zeros((n_genomes,n_param))

    for j in range(2): # We need two rounds!
        shuffle = np.random.permutation(total_population.shape[0])
        total_population = total_population[shuffle,:]
        fitness = fitness[shuffle]



        for i in range(int(2*n_genomes / 4)):
            fitnesses = fitness[[4*i, 4*i+1, 4*i+2, 4*i+3]]
            best_index = np.argmin(fitnesses)

            new_population[int(j*n_genomes/2 + i)] = total_population[4*i + best_index,:]



    
    population = new_population
    print('Best score:', np.min(fitness), np.mean(fitnesses), time.time() - start_time, 's')
    time.sleep(0.01) # for ctrl+c
    epoch += 1
    start_time = time.time()