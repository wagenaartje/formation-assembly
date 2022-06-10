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
print(np.argmin(fitnesses))

# Simulate once
N = 10000
population = np.repeat(best_genome, N, axis=0)


config['dt'] = 0.001
config['t_max'] = 10

f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)



print('f: ', np.mean(f1s+f2s))
print('f1:', np.mean(f1s))
print('f2:', np.mean(f2s))
print('Collision rate:', np.sum(collided==0) / N * 100 ,'%')
print('Success rate:', np.sum(f1s<0.1) / N * 100, '%')

non_collided = collided == 1
f1s = f1s[non_collided]
print('Success rate ignoring collisions:', np.sum(f1s<0.1) / f1s.size * 100, '%')