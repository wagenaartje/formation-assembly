# What if we do a formation that is all zeros? Will they avoid or collide.

import sys
sys.path.append('.')

import itertools
import json
import numpy as np
from evaluation import gen_points_min_dist, population_action
import matplotlib.pyplot as plt



def single_evaluate(config: dict, population: np.ndarray, analysis: bool = False) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population on a single evaluation'''
    permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))

    loops = int(config['t_max']/config['dt'])

    # Initialize random initial positions and formations
    initial_position = gen_points_min_dist(config)
    
    positions = np.zeros((population.shape[0],3,2))
    for i in range(population.shape[0]):
        positions[i] = gen_points_min_dist(config)

    formation = np.reshape(np.asarray([[-1,0], [1,0], [0,np.sqrt(3)]]), (1,3,2))
    formation_input = np.repeat(formation.copy(),population.shape[0],axis=0)
 
    
    formation_input -= np.mean(formation_input,axis=1,keepdims=True)

    # If save=true, log the position history
    if analysis: position_history = np.zeros((loops, population.shape[0], config['n_agents'], 2))

    velocity = np.zeros((population.shape[0],config['n_agents'],2)) 
    collided = np.ones(population.shape[0])

    # Now, we have to go over all possible combinations and take the minimum
    for i in range(loops):
        inputs = np.zeros((population.shape[0],0,config['n_inputs']))
        
        # NOTE! We must keep this here, since we subtract formation_input and positions later.
        positions_centered = positions - np.mean(positions,axis=1,keepdims=True)
        

        if analysis: position_history[i] = positions.copy()

        for j in range(config['n_agents']):
            # Gather inputs for 1st agent
            agents = np.delete(np.arange(config['n_agents']),j)
            np.random.shuffle(agents)

            formation_select = np.arange(config['n_agents'])
            np.random.shuffle(formation_select)

            inputs_0 = positions_centered[:,agents,:] - positions_centered[:,[j],:]

            distances = np.min(np.linalg.norm(inputs_0,axis=2),axis=1)
            collided = np.where(distances < config[ 'd_min'], np.zeros(population.shape[0]), collided)

            inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(config['n_agents']-1)*2))

            formation_enter = formation_input[:,formation_select,:] - positions_centered[:,[j],:]
            formation_enter = np.reshape(formation_enter, (population.shape[0], 1,config['n_agents']*2))
            inputs_0 = np.concatenate((velocity[:,[j],:], inputs_0, formation_enter), axis=2)


            # Concatenate to 3 samples per genome
            inputs = np.concatenate((inputs,inputs_0),axis=1)
        
        # Get action
        acceleration = population_action(config, population, inputs) * config['a_max']

        acceleration_magnitude = np.linalg.norm(acceleration,axis=2,keepdims=True)
        acceleration = np.where(acceleration_magnitude < config['a_max'], acceleration, acceleration / acceleration_magnitude * config[ 'a_max'])

        velocity += acceleration * config['dt']


        # Clip the velocity
        velocity_magnitude = np.linalg.norm(velocity,axis=2,keepdims=True)
        velocity = np.where(velocity_magnitude < config['v_max'], velocity, velocity / velocity_magnitude * config['v_max'])

        # Update the position (if not collided!)
        collided_full = np.reshape(collided, (population.shape[0],1,1))
        positions += velocity * config['dt']

    # Now at the end, compare to formation
    positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    #formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    f1 = np.ones(population.shape[0]) * np.inf
    for order in permutations:
        rel_locations = positions_c[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_input,axis=2),axis=1)

        f1 = np.where(rel_dist_diff < f1, rel_dist_diff, f1) 

    # Boundary conditions
    f2 = np.mean(np.linalg.norm(velocity,axis=2),axis=1)

    if not analysis:
        return f1 + f2
    else:
        return f1, f2, formation_input, position_history, collided 


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


    return best_order_index


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


config['dt'] = 0.01
#config['t_max'] = 10

f1s, f2s, formation, position_history, collided = single_evaluate(config, population, analysis=True)

print(np.mean(f1s + f2s))
print(np.sum(collided == 0) / N)

''' Formation plot '''
fig1, ax1 = plt.subplots(1)
ax1.scatter(formation[0,:,0], formation[0,:,1],label='Target',s=60,c='white',edgecolor='black')
ax1.set_aspect(1)

ax1.set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=12)
ax1.set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=12)
plt.locator_params(nbins=5)


position_history -= np.mean(position_history[[-1],:,:,:], axis=2, keepdims=True)

true_indexing = optimal_index(position_history[-1,:,:,:], formation).astype('int')
permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))

'''  Positions plot'''
fig2, ax2 = plt.subplots(1)

colors = ['r', 'g', 'b']

for j in range(N):
    for i in range(config['n_agents']):
        ax2.plot(position_history[:,j,i,0], position_history[:,j,i,1],colors[permutations[true_indexing[j]][i]], alpha=0.1)

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



plt.show()