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

population = np.repeat(best_genome, N, axis=0)

#dts = np.asarray([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128])
#t_max = np.linspace(2,10,8)
dts = np.asarray([0.001, 0.005, 0.01, 0.05, 0.1])
t_max = np.linspace(2,10,9)

results = np.zeros((dts.size, t_max.size, 3))

for i in range(dts.size):
    for j in range(t_max.size):
        config['dt'] = dts[i]
        config['t_max'] = t_max[j]
        f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

        results[i,j,0] = np.mean(f1s+f2s)
        results[i,j,1] = np.sum(collided == 0) / N 

        f1s = f1s[collided == 1]
        results[i,j,2] = np.sum(f1s < 0.1) / f1s.size

''' f heatmap'''
fig, ax = plt.subplots(1)
X,Y=np.meshgrid(dts,t_max)
c = sns.heatmap(results[:,:,0].T, cmap='viridis',xticklabels=dts, yticklabels=t_max, annot=True, vmin=0,cbar=False)
plt.xlabel(r'$\Delta t\ (\mathrm{s})$',fontsize=12)
plt.ylabel(r'$T\ (\mathrm{s})$', fontsize=12)
ax.invert_yaxis()
ax.set_yticklabels(t_max.astype('int'))

ax.set_box_aspect(1)

plt.savefig('./figures/f_heatmap.png', dpi=300, bbox_inches='tight')

''' collided heatmap'''
fig, ax = plt.subplots(1)
X,Y=np.meshgrid(dts,t_max)
c = sns.heatmap(results[:,:,1].T, cmap='viridis',xticklabels=dts, yticklabels=t_max, annot=True, vmin=0, fmt='.1%',cbar=False)
plt.xlabel(r'$\Delta t\ (\mathrm{s})$',fontsize=12)
plt.ylabel(r'$T\ (\mathrm{s})$', fontsize=12)
ax.invert_yaxis()
ax.set_yticklabels(t_max.astype('int'))

ax.set_box_aspect(1)

plt.savefig('./figures/c_heatmap.png', dpi=300, bbox_inches='tight')

''' success heatmap'''
fig, ax = plt.subplots(1)
X,Y=np.meshgrid(dts,t_max)
c = sns.heatmap(results[:,:,2].T, cmap='viridis',xticklabels=dts, yticklabels=t_max, annot=True, vmin=0, fmt='.1%',cbar=False)
plt.xlabel(r'$\Delta t\ (\mathrm{s})$',fontsize=12)
plt.ylabel(r'$T\ (\mathrm{s})$', fontsize=12)
ax.invert_yaxis()
ax.set_yticklabels(t_max.astype('int'))

ax.set_box_aspect(1)

plt.savefig('./figures/s_heatmap.png', dpi=300, bbox_inches='tight')


plt.show()
