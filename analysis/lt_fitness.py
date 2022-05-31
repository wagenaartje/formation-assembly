import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
from settings import *
from evaluation import evaluate_population

n_skip = 1 # how much genomes to skip before evaluating the next one
n_windows = 20 # must be even

# Load history
genomes = np.fromfile('./output/genome.dat')
genomes = np.reshape(genomes, (-1, n_param))
epochs = np.arange(1,genomes.shape[0]+1)

# Skip every n_skip-th element to reduce computation time
genomes = genomes[::n_skip]
epochs = epochs[::n_skip]

# Simulate the genome
lt_fitnesses,bcs = evaluate_population(genomes, n_steps, lt_fitness=True)

# Convolve
lt_fitnesses = np.convolve(lt_fitnesses, np.ones(n_windows) / n_windows, mode='valid')


''' Long-term fitness history '''
fig, ax = plt.subplots(1)
if n_windows != 1:
    ax.plot(epochs[int(n_windows/2)-1:-int(n_windows/2)],-lt_fitnesses)
else:
    ax.plot(epochs,-lt_fitnesses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean distance to formation')
ax.set_title('Long-term fitness with moving mean of {0}'.format(n_windows))

plt.show()