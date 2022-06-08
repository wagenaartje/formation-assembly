import sys
sys.path.append('.')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import settings
from evaluation import single_evaluate

# NOTE to self: settings should be loaded from the str here?

#file_name = 'agents=3hidden=8evals=10steps=100pop=100pc=0.5pm=0.05.dat'
#run_settings = settings.from_file(file_name)

file_name = settings.to_str() + '.dat'
run_settings = settings.from_file(file_name)

# Recover some other settings
n_hidden = run_settings['n_hidden']
n_inputs = (run_settings['n_agents'])*4  # Input neurons
n_outputs = 2               # Output neurons
n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs

# Load best genomes and their fitnesses
genomes = np.fromfile('./runs/g_' + file_name)
genomes = np.reshape(genomes, (-1, n_param))
fitnesses = np.fromfile('./runs/f_' + file_name)

# Select the last genome
best_genome = genomes[[-1],:]

# Simulate once
fitness, bcs = single_evaluate(best_genome, run_settings['t_max'], True, True)


# Load the results
initial_position = np.load('./tmp/initial_position.npy')
formation = np.load('./tmp/formation.npy')
position_history = np.load('./tmp/position_history.npy')

print(position_history.shape)

print('Total acceleration:', np.sum(np.mean(np.linalg.norm(np.diff(position_history,n=2,axis=0),axis=3),axis=2))  /0.05)

print(bcs)

print('Fitness:', fitness)

# Plot the results
fig, axes = plt.subplots(1,3)

axes[0].scatter(formation[0,:,0], formation[0,:,1],label='Target')

axes[0].scatter(initial_position[0,:,0], initial_position[0,:,1],label='Start')

com = np.mean(position_history,axis=2)
axes[0].scatter(com[0,0,0], com[0,0,1])
axes[0].scatter(com[-1,0,0], com[-1,0,1])

for i in range(run_settings['n_agents']):
    axes[0].plot(position_history[:,0,i,0], position_history[:,0,i,1],':')


axes[0].scatter(position_history[-1,0,:,0], position_history[-1,0,:,1],c='red',label='End')

axes[0].set_aspect(1)

rect = Rectangle((-2,-2),4,4,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
axes[0].add_patch(rect)

axes[0].legend()



''' Velocity & acceleration '''
velocity = np.linalg.norm(np.diff(position_history,n=1,axis=0),axis=3)
time = run_settings['dt']* np.arange(velocity.shape[0])

print(velocity.shape)

for i in range(run_settings['n_agents']):
    axes[1].plot(time,velocity[:,0,i] / run_settings['dt'])


acceleration = np.linalg.norm(np.diff(position_history,n=2,axis=0),axis=3)
time = run_settings['dt']* np.arange(acceleration.shape[0])

for i in range(run_settings['n_agents']):
    axes[2].plot(time,acceleration[:,0,i] / run_settings['dt']**2)





plt.show()
