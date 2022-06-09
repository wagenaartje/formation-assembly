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
dts = np.asarray([0.001, 0.005, 0.01, 0.05, 0.1])
t_max = np.linspace(2,10,9)

results = np.zeros((dts.size, t_max.size))

for i in range(dts.size):
    for j in range(t_max.size):
        config['dt'] = dts[i]
        config['t_max'] = t_max[j]
        f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

        results[i,j] = np.mean(f1s)

fig, ax = plt.subplots(1)
#plt.imshow(results.T, extent=[dts[0], dts[-1], t_max[0], t_max[-1]], origin='lower', vmin=0, aspect=(dts[-1]-dts[0]) / (t_max[-1] - t_max[0])) # interpolation='Gaussian'
X,Y=np.meshgrid(dts,t_max)
c = sns.heatmap(results.T, cmap='viridis',xticklabels=dts, yticklabels=t_max, annot=True, vmin=0)
# plt.colorbar()
plt.xlabel(r'$\Delta t\ (\mathrm{s})$',fontsize=12)
plt.ylabel(r'$T\ (\mathrm{s})$', fontsize=12)
ax.invert_yaxis()
#plt.xscale('log')
ax.set_yticklabels(t_max.astype('int'))

# def show_values(pc, **kw):
#     global ax
#     pc.update_scalarmappable()
#     for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
#         x, y = p.vertices[:-2, :].mean(0)
#         if np.all(color[:3] > 0.5):
#             color = (0.0, 0.0, 0.0)
#         else:
#             color = (1.0, 1.0, 1.0)
#         ax.text(x, y, '%.2f' % (value), ha="center", va="center", color=color, **kw)

# show_values(c)

ax.set_box_aspect(1)

plt.savefig('./figures/f1_time_new.png', dpi=300)

# NOTE! We should do interpolation

plt.show()
