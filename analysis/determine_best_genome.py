import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate
import seaborn as sns

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

# Simulate once
N = 1000

best_fitness = np.inf
best_index = None
for i in range(100):
    population = np.repeat(genomes[[-i-1],:], N, axis=0)
    f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

    fitness = np.mean(f1s + f2s)
    print(i, fitness)
    if fitness < best_fitness:
        best_fitness = fitness
        best_index = -i-1

print()
print('==== RESULTS ====')
print(best_index, best_fitness)