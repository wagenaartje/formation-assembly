import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate
import itertools


run = './results/base/1654698799/'

fitnesses = np.fromfile(run + 'fitnesses.dat')
genomes = np.fromfile(run + 'genomes.dat').reshape((fitnesses.shape[0], -1))
with open(run + 'config.json') as f:
    config = json.load(f)

config['dt'] = 0.01
config['t_max'] = 5

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

distance = np.mean(np.sum(np.linalg.norm(np.gradient(position_history, axis=0),axis=3),axis=0),axis=1)

positions_c = position_history - np.mean(position_history, axis=2, keepdims=True)
permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))
f1 = np.ones((position_history.shape[0], population.shape[0])) * np.inf
for order in permutations:
    rel_locations = positions_c[:,:,list(order),:]
    rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation,axis=3),axis=2)

    f1 = np.where(rel_dist_diff < f1, rel_dist_diff, f1)

times = np.argmax(f1 < 0.1,axis=0) * config['dt']

# Remove the cases where not reached at all.
keep = times != 0
times = times[keep]
distance = distance[keep]

print(np.mean(f1s), np.mean(f2s))

print(np.sum(keep.astype('int')) / population.shape[0] * 100, '%')


print(np.sum(collided) / collided.size * 100, '%')

t_edges = np.linspace(0,config['t_max'],25)
d_edges = np.linspace(0,3,25)

heatmap,xedges,yedges = np.histogram2d(times,distance, bins=(t_edges,d_edges))
fig,ax = plt.subplots(1)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin='lower',aspect=t_edges[-1] / d_edges[-1])

plt.show()
