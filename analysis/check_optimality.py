import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate
import itertools

run = './results/base/1654760251/'

fitnesses = np.fromfile(run + 'fitnesses.dat')
genomes = np.fromfile(run + 'genomes.dat').reshape((fitnesses.shape[0], -1))
with open(run + 'config.json') as f:
    config = json.load(f)



# Add related parameters
config['n_inputs'] = config['n_agents']*4 # Input neurons
config['n_outputs'] = 2                   # Output neurons

# Total number of parameters
config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']


# Select the last genome
best_genome = genomes[[np.argmin(fitnesses)],:]
print(np.argmin(fitnesses))

# Simulate once
N = 10000
population = np.repeat(best_genome, N, axis=0)


config['dt'] = 0.001
config['t_max'] = 10

f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)



def optimal_index (initial_positions, formations):
    permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))

    centered_positions = initial_positions - np.mean(initial_positions,axis=1,keepdims=True)
    centered_formations = formations - np.mean(formations,axis=1, keepdims=True)



    best_diff = np.ones(population.shape[0]) * np.inf
    best_order_index = np.zeros(population.shape[0])
    for i in range(len(permutations)):
        order = permutations[i]
        # permute formation, and see for which relative distances is most similar.
        formation_copy = centered_formations[:,list(order),:].copy()

        rel_dist_diff = np.max(np.linalg.norm(centered_positions - formation_copy,axis=2),axis=1)


        best_order_index = np.where(rel_dist_diff < best_diff, np.ones(population.shape[0]) * i, best_order_index)
        best_diff = np.where(rel_dist_diff < best_diff, rel_dist_diff, best_diff)


    return best_order_index, best_diff




optimal_indexing, minimal_max = optimal_index(position_history[0,:,:,:], formation)
true_indexing, _ = optimal_index(position_history[-1,:,:,:], formation)
true_indexing = true_indexing.astype('int')

permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))

formation = formation  - np.mean(formation,axis=1,keepdims=True)+ np.mean(position_history[0,:,:,:],axis=1,keepdims=True)

for i in range(N):
    formation[i,:,:] = formation[i,permutations[true_indexing[i]], :]

#true_max = np.max(np.linalg.norm(formation[:,permutations[true_indexing[],:] - position_history[0,:,:,:],axis=2),axis=1)

true_max = np.max(np.linalg.norm(formation - position_history[0,:,:,:],axis=2),axis=1)

true_max = np.around(true_max, 2)
minimal_max = np.around(minimal_max, 2)

print(np.sum(minimal_max == true_max) / N * 100, '%')

print(np.sum(optimal_indexing == true_indexing) / N * 100, '%')

# Now compare with the true indexing