import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate

save_number = 'collision'

run = './results/base/1654760251/'

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
best_genome = genomes[[np.argmin(fitnesses)],:]

# Simulate once
if save_number == 'collision':
    collided = [1]

    while collided[0] == 1:
        f1, f2, formation, position_history, collided = single_evaluate(config, best_genome, analysis=True)


if save_number == 'failure':
    f1 = 0.05
    collided = [1]

    while f1 < 0.1 or collided[0] == 0:
        f1, f2, formation, position_history, collided = single_evaluate(config, best_genome, analysis=True)

if save_number == 'success':
    f1, f2, formation, position_history, collided = single_evaluate(config, best_genome, analysis=True)

print('Fitness:', f1, f2, collided)

# Plot the results
''' Formation plot '''
fig1, ax1 = plt.subplots(1)
ax1.scatter(formation[0,:,0], formation[0,:,1],label='Target',s=60,c='white',edgecolor='black')
ax1.set_aspect(1)

ax1.set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=12)
ax1.set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=12)
plt.locator_params(nbins=5)

'''  Positions plot'''
fig2, ax2 = plt.subplots(1)
for i in range(config['n_agents']):
    ax2.plot(position_history[:,0,i,0], position_history[:,0,i,1],'k:')
    ax2.text(position_history[-1,0,i,0]+0.05, position_history[-1,0,i,1]+0.07, f'{i}')

ax2.scatter(position_history[0,0,:,0], position_history[0,0,:,1],c='white',label='Start',edgecolor='black', linewidth=1, s=60,hatch='///////',zorder=10)
ax2.scatter(position_history[-1,0,:,0], position_history[-1,0,:,1],c='white',label='End',edgecolor='black', linewidth=1, s=60,zorder=10)

xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
xsize = np.diff(xlim)
xcent = np.mean(xlim)
ysize = np.diff(ylim)
ycent = np.mean(ylim)

if xsize > ysize:
    ax2.set_ylim([ycent - xsize/2, ycent + xsize/2])
    biggest = xsize
else:
    ax2.set_xlim([xcent - ysize/2, xcent + ysize/2])
    biggest = ysize

xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
xcent = np.mean(xlim)
ycent = np.mean(ylim)

ax1.set_xlim([xcent - biggest/2, xcent + biggest/2])
ax1.set_ylim([ycent - biggest/2, ycent + biggest/2])

ax2.set_aspect(1)
ax2.legend()
ax2.set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=12)
ax2.set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=12)
plt.locator_params(nbins=5)


''' Velocity '''

fig3, ax3 = plt.subplots(1)
velocity = np.linalg.norm(np.diff(position_history,n=1,axis=0),axis=3) / config['dt']
time = config['dt']* np.arange(velocity.shape[0])

if save_number == 'collision':
    checker = np.min(velocity,axis=2)[:,0]
    time = time[checker != 0]
    velocity = velocity[checker != 0]


print(velocity.shape)

linestyles = ['-', '--', ':']
for i in range(config['n_agents']):
    ax3.plot(time,velocity[:,0,i], 'k', alpha=1-i/3, label=str(i), linewidth=1)

ax3.set_box_aspect(1)
ax3.legend()

ax3.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=12)
ax3.set_ylabel(r'$v\ \mathrm{(m s^{-1})}$', fontsize=12)
ax3.set_xlim([0,time[-1]])
ax3.set_ylim([0, 1.1])
plt.locator_params(nbins=5)


fig1.savefig('./figures/runs/{0}_f.png'.format(save_number), bbox_inches='tight', dpi=300)
fig2.savefig('./figures/runs/{0}_p.png'.format(save_number), bbox_inches='tight', dpi=300)
fig3.savefig('./figures/runs/{0}_v.png'.format(save_number), bbox_inches='tight', dpi=300)

plt.show()
