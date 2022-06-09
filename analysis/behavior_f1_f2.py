import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate


run = './results/base/1654698799/'

fitnesses = np.fromfile(run + 'fitnesses.dat')
genomes = np.fromfile(run + 'genomes.dat').reshape((fitnesses.shape[0], -1))
with open(run + 'config.json') as f:
    config = json.load(f)

config['dt'] = 0.01

# Add related parameters
config['n_inputs'] = config['n_agents']*4 # Input neurons
config['n_outputs'] = 2                   # Output neurons

# Total number of parameters
config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']


# Select the last genome
best_genome = genomes[[-1],:]

# Simulate once
N = 10000

population = np.repeat(best_genome, N, axis=0)


f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

print(np.sum(collided) / collided.size * 100, '%')

x_edges = np.linspace(0,0.15,25)
y_edges = np.linspace(0,0.15,25)

heatmap,xedges,yedges = np.histogram2d(f1s,f2s, bins=(x_edges,y_edges))
fig,ax = plt.subplots(1)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin='lower')

# NOTE! Due to outliers, the median/mean are far from the "hot" parts

ax.set_xlabel(r'$f_1$',fontsize=12)
ax.set_ylabel(r'$f_2$',fontsize=12)

print(np.mean(f1s), np.mean(f2s))

# NOTE! Increasing dt has little effect on collision avoidance for low dt.

plt.show()
