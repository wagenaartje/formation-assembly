import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate
import seaborn as sns
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
# Simulate once
N = 100
population = np.repeat(best_genome, N, axis=0)


#config['dt'] = 0.001
#config['t_max'] = 10

f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

position_history = position_history[:,collided==1,:,:]

pairs = np.asarray(list(itertools.combinations(np.arange(config['n_agents']), 2)))
print(position_history.shape)


differences = position_history[:,:,pairs[:,0],:] - position_history[:,:,pairs[:,1],:]
distances = np.linalg.norm(differences,axis=3)

minimum_distance = np.min(distances,axis=(0,2))

print(np.mean(minimum_distance))


