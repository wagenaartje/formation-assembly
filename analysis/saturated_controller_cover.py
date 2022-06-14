import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import json
from evaluation import single_evaluate
from scipy.ndimage.filters import gaussian_filter

run = './results/base/1654760251/'

fitnesses = np.fromfile(run + 'fitnesses.dat')
genomes = np.fromfile(run + 'genomes.dat').reshape((fitnesses.shape[0], -1))
with open(run + 'config.json') as f:
    config = json.load(f)

config['dt'] = 0.1

# Add related parameters
config['n_inputs'] = config['n_agents']*4 # Input neurons
config['n_outputs'] = 2                   # Output neurons

# Total number of parameters
config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']


# Select the last genome
best_genome = genomes[[np.argmin(fitnesses)],:]

# Simulate once
N = 10000

population = np.repeat(best_genome, N, axis=0)
f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)


#print(bcs)

# Plot the results
fig, ax = plt.subplots(1)




''' Velocity & acceleration '''
acceleration_c = np.diff(position_history,n=2,axis=0) / config['dt']**2

print(acceleration_c.shape)

print(acceleration_c.shape)
acceleration_c = np.reshape(acceleration_c, (-1, 2))

keep = np.linalg.norm(acceleration_c, axis=1) <= config['a_max']
acceleration_c = acceleration_c[keep]

xedges = np.linspace(-2.2, 2.2, 100)
yedges = np.linspace(-2.2, 2.2, 100)

heatmap, xedges, yedges = np.histogram2d(acceleration_c[:,0], acceleration_c[:,1], bins=(xedges, yedges))
heatmap /= np.sum(heatmap)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


heatmap = gaussian_filter(heatmap, sigma=0.5)
#plt.pcolor(heatmap.T,norm=matplotlib.colors.LogNorm(), cmap='jet')
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet',norm=matplotlib.colors.LogNorm(),vmin=10e-7)

ax.set_box_aspect(1)
plt.axis('off')

plt.savefig('./figures/a_heat_map.png'.format(config['dt']), bbox_inches='tight')

plt.show()
