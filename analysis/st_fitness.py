import matplotlib.pyplot as plt
import numpy as np

fitnesses = np.fromfile('./output/fitness.dat')
epochs = np.arange(1,fitnesses.shape[0]+1)

fig, ax = plt.subplots(1)

''' Fitness history '''
n_windows = 20 # must be even
fitnesses = np.convolve(fitnesses, np.ones(n_windows) / n_windows, mode='valid')


if n_windows != 1:
    ax.plot(epochs[int(n_windows/2)-1:-int(n_windows/2)],-fitnesses)
else:
    ax.plot(epochs,-fitnesses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Fitness')
ax.set_title('Best fitness with moving mean of {0}'.format(n_windows))
ax.grid()


plt.show()