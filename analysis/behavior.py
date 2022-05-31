import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import settings
from evaluation import single_evaluate

# NOTE to self: settings should be loaded from the str here?

file_name = 'agents=3hidden=8evals=10steps=100pop=100pc=0.5pm=0.05.dat'
run_settings = settings.from_file(file_name)

# Recover some other settings
n_hidden = run_settings['n_hidden']
n_inputs = (run_settings['n_agents'])*4 -2  # Input neurons
n_outputs = 2               # Output neurons
n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs

# Load best genomes and their fitnesses
genomes = np.fromfile('./runs/g_' + file_name)
genomes = np.reshape(genomes, (-1, n_param))
fitnesses = np.fromfile('./runs/f_' + file_name)

# Select the last genome
best_genome = genomes[[-1],:]

# Simulate once
single_evaluate(best_genome, run_settings['n_steps'], True, True)

# Load the results
initial_position = np.load('./tmp/initial_position.npy')
formation = np.load('./tmp/formation.npy')
position_history = np.load('./tmp/position_history.npy')

print(position_history.shape)

print('Total acceleration:', np.sum(np.mean(np.linalg.norm(np.diff(position_history,n=2,axis=0),axis=3),axis=2)) / run_settings['n_steps'] /0.05)


# Plot the results
fig, ax = plt.subplots(1)
formation -= np.mean(formation,axis=1, keepdims=True)
ax.scatter(formation[0,:,0], formation[0,:,1],label='Target')

initial_position -= np.mean(initial_position,axis=1, keepdims=True)
ax.scatter(initial_position[0,:,0], initial_position[0,:,1],label='Start')


for i in range(run_settings['n_agents']):
    ax.plot(position_history[:,0,i,0], position_history[:,0,i,1],':')


ax.scatter(position_history[-1,0,:,0], position_history[-1,0,:,1],c='red',label='End')

ax.set_aspect(1)

plt.legend()
plt.show()
