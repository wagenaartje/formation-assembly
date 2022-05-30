import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
from settings import *
from evaluation import single_evaluate

# Load best genomes and their fitnesses
genomes = np.fromfile('./output/genome.dat')
genomes = np.reshape(genomes, (-1, n_param))
fitnesses = np.fromfile('./output/fitness.dat')
print(np.min(fitnesses))

# Select the genome with the best fitness
best_genome = genomes[[-1],:]

# Simulate once
single_evaluate(best_genome, n_steps_lt, True, True)

# Load the results
initial_position = np.load('./output/initial_position.npy')
formation = np.load('./output/formation.npy')
position_history = np.load('./output/position_history.npy')


# Plot the results
fig, ax = plt.subplots(1)
formation -= np.mean(formation,axis=1, keepdims=True)
ax.scatter(formation[0,:,0], formation[0,:,1],label='Target')

initial_position -= np.mean(initial_position,axis=1, keepdims=True)
ax.scatter(initial_position[0,:,0], initial_position[0,:,1],label='Start')


for i in range(n_agents):
    ax.plot(position_history[:,0,i,0], position_history[:,0,i,1],':')


ax.scatter(position_history[-1,0,:,0], position_history[-1,0,:,1],c='red',label='End')

ax.set_aspect(1)

plt.legend()
plt.show()
