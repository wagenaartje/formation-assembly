import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import os
import settings



fig, ax = plt.subplots(1)


file_name = 'f_' + settings.to_str() + '.dat'
folder = './runs/'

run_settings = settings.from_file(file_name)


fitnesses = np.fromfile(folder + file_name)
epochs = np.arange(1,fitnesses.shape[0]+1)



''' Fitness history '''
n_windows = 20 # must be even
fitnesses = np.convolve(fitnesses, np.ones(n_windows) / n_windows, mode='valid')


if n_windows != 1:
    ax.plot(epochs[int(n_windows/2)-1:-int(n_windows/2)],-fitnesses,label=str(run_settings['n_hidden']))
else:
    ax.plot(epochs,-fitnesses,label=str(run_settings['n_hidden']))

ax.set_xlabel('Epoch')
ax.set_ylabel('Fitness')
ax.set_title('Best fitness with moving mean of {0}'.format(n_windows))
ax.grid()
ax.legend()


plt.show()