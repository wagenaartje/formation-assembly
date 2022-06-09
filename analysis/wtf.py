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
best_genome = genomes[[-1],:]

# Simulate once
N = 1000

population = np.repeat(best_genome, N, axis=0)

#dts = np.asarray([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128])
#t_max = np.linspace(2,10,8)
dts = np.asarray([0.01])
t_max = np.asarray([5])

results = np.zeros((dts.size, t_max.size))

for i in range(dts.size):
    for j in range(t_max.size):
        config['dt'] = dts[i]
        config['t_max'] = t_max[j]
        f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

        results[i,j] = np.mean(f1s+f2s)
        print(np.mean(f1s+f2s), np.mean(f1s), np.mean(f2s))

        keep = collided == 1
        f1s = f1s[keep]
        f2s = f2s[keep]

        print(np.mean(f1s+f2s), np.mean(f1s), np.mean(f2s))


