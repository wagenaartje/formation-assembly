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

# Add related parameters
config['n_inputs'] = config['n_agents']*4 # Input neurons
config['n_outputs'] = 2                   # Output neurons

# Total number of parameters
config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']


# Select the last genome
best_genome = genomes[[-1],:]

# Simulate once
fitness  = single_evaluate(config, best_genome, save=True)


# Load the results
initial_position = np.load('./tmp/initial_position.npy')
formation = np.load('./tmp/formation.npy')
position_history = np.load('./tmp/position_history.npy')

print(position_history.shape)

print('Total acceleration:', np.sum(np.mean(np.linalg.norm(np.diff(position_history,n=2,axis=0),axis=3),axis=2))  /0.05)

#print(bcs)

print('Fitness:', fitness)

# Plot the results
fig, axes = plt.subplots(1,3)

axes[0].scatter(formation[0,:,0], formation[0,:,1],label='Target')

axes[0].scatter(initial_position[0,:,0], initial_position[0,:,1],label='Start')

com = np.mean(position_history,axis=2)
axes[0].scatter(com[0,0,0], com[0,0,1])
axes[0].scatter(com[-1,0,0], com[-1,0,1])

for i in range(config['n_agents']):
    axes[0].plot(position_history[:,0,i,0], position_history[:,0,i,1],':')


axes[0].scatter(position_history[-1,0,:,0], position_history[-1,0,:,1],c='red',label='End')

axes[0].set_aspect(1)




axes[0].legend()



''' Velocity & acceleration '''
velocity = np.linalg.norm(np.diff(position_history,n=1,axis=0),axis=3)
time = config['dt']* np.arange(velocity.shape[0])

print(velocity.shape)

for i in range(config['n_agents']):
    axes[1].plot(time,velocity[:,0,i] / config['dt'])


acceleration = np.linalg.norm(np.diff(position_history,n=2,axis=0),axis=3)
time = config['dt']* np.arange(acceleration.shape[0])

for i in range(config['n_agents']):
    axes[2].plot(time,acceleration[:,0,i] / config['dt']**2)





plt.show()
