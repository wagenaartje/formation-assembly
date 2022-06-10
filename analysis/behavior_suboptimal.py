import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate
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
print(np.argmin(fitnesses))

# Simulate once
N = 1
population = np.repeat(best_genome, N, axis=0)


config['dt'] = 0.001
config['t_max'] = 5





def optimal_index (initial_positions, formations):
    permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))

    centered_positions = initial_positions - np.mean(initial_positions,axis=1,keepdims=True)
    centered_formations = formations - np.mean(formations,axis=1, keepdims=True)



    best_diff = np.ones(population.shape[0]) * np.inf
    best_order_index = np.zeros(population.shape[0])
    for i in range(len(permutations)):
        order = permutations[i]
        # permute formation, and see for which relative distances is most similar.
        formation_copy = centered_formations[:,list(order),:].copy()

        rel_dist_diff = np.max(np.linalg.norm(centered_positions - formation_copy,axis=2),axis=1)


        best_order_index = np.where(rel_dist_diff < best_diff, np.ones(population.shape[0]) * i, best_order_index)
        best_diff = np.where(rel_dist_diff < best_diff, rel_dist_diff, best_diff)


    return best_order_index, best_diff

permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))
keep_going = True

while keep_going:
    f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True, identical_start=False)

    optimal_indexing, minimal_max = optimal_index(position_history[0,:,:,:], formation)
    true_indexing, _ = optimal_index(position_history[-1,:,:,:], formation)
    true_indexing = true_indexing.astype('int')

    formation_copy = formation[:,permutations[true_indexing[0]], :]
    
    formation_copy = formation_copy  - np.mean(formation_copy,axis=1,keepdims=True)+ np.mean(position_history[0,:,:,:],axis=1,keepdims=True)

    true_max = np.max(np.linalg.norm(formation_copy - position_history[0,:,:,:],axis=2),axis=1)

    true_max = np.around(true_max, 2)
    minimal_max = np.around(minimal_max, 2)

    #keep_going = (optimal_indexing == true_indexing).all()


    keep_going = (true_max == minimal_max).all() or f1s > 0.1

print(true_max, minimal_max)

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
    #ax2.text(position_history[-1,0,i,0]+0.05, position_history[-1,0,i,1]+0.07, f'{i}')

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

fig4, ax4 = plt.subplots(1)
velocity = np.linalg.norm(np.diff(position_history,n=1,axis=0),axis=3) / config['dt']
time = config['dt']* np.arange(velocity.shape[0])



print(velocity.shape)

linestyles = ['-', '--', ':']
for i in range(config['n_agents']):
    ax4.plot(time,velocity[:,0,i], 'k', alpha=1-i/3, label=str(i), linewidth=1)

ax4.set_box_aspect(1)
ax4.legend()

ax4.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=12)
ax4.set_ylabel(r'$v\ \mathrm{(m s^{-1})}$', fontsize=12)
ax4.set_xlim([0,time[-1]])
ax4.set_ylim([0, 1.1])
plt.locator_params(nbins=5)

''' Estimated optimal assignment '''
fig3, ax3 = plt.subplots(1)

ax3.scatter(position_history[0,0,:,0], position_history[0,0,:,1],c='white',label='Start',edgecolor='black', linewidth=1, s=60,hatch='///////',zorder=10)


formation = formation - np.mean(formation,axis=1,keepdims=True) + np.mean(position_history[0,:,:,:], axis=1, keepdims=True)

ax3.scatter(formation[0,:,0], formation[0,:,1],c='white',label='End',edgecolor='black', linewidth=1, s=60,zorder=10)

permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))
optimal_indexing = permutations[int(optimal_indexing[0])]
for i in range(config['n_agents']):
    ax3.plot([position_history[0,0,i,0], formation[0,optimal_indexing[i],0]], [position_history[0,0,i,1], formation[0,optimal_indexing[i],1]], 'k:')

xlim = ax3.get_xlim()
ylim = ax3.get_ylim()
xcent = np.mean(xlim)
ycent = np.mean(ylim)

ax3.set_xlim([xcent - biggest/2, xcent + biggest/2])
ax3.set_ylim([ycent - biggest/2, ycent + biggest/2])
ax3.set_aspect(1)
ax3.legend()
ax3.set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=12)
ax3.set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=12)
plt.locator_params(nbins=5)

#fig1.savefig('./figures/suboptimal/f.png', bbox_inches='tight', dpi=300)
#fig2.savefig('./figures/suboptimal/p_true.png', bbox_inches='tight', dpi=300)
#fig3.savefig('./figures/suboptimal/p_opt.png', bbox_inches='tight', dpi=300)


plt.show()