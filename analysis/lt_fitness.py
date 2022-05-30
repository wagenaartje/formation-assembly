import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
from settings import *
from evaluation import evaluate_population

n_skip = 50 # how much genomes to skip before evaluating the next one

genomes = np.fromfile('./output/genome.dat')
genomes = np.reshape(genomes, (-1, n_param))
epochs = np.arange(1,genomes.shape[0]+1)

genomes = genomes[::n_skip]
epochs = epochs[::n_skip]


fig, ax = plt.subplots(1)

lt_fitnesses = evaluate_population(genomes, n_steps_lt, True)



''' Long-term fitness history '''

ax.plot(epochs,-lt_fitnesses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean distance to formation')
ax.set_title('Long-term fitness')

plt.show()