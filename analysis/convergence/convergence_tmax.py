import numpy as np
import json
import os
import matplotlib.pyplot as plt

folder = '/tmax/'
key = 't_max'

fig,ax = plt.subplots(1)

epochs = np.arange(5001)

all_fitnesses = {}

subfolders = os.listdir('./results/' + folder)
for subfolder in subfolders:
    fitnesses = np.fromfile('./results/' + folder +  '/' + subfolder + '/fitnesses.dat')
    with open('./results/' + folder +  '/' + subfolder + '/config.json') as f:
        config = json.load(f)


    if fitnesses.shape[0] == epochs.shape[0]:
        if config[key] not in all_fitnesses:
            all_fitnesses[config[key]] = []
        all_fitnesses[config[key]].append(fitnesses)

print(len(all_fitnesses))

n_windows = 100
epochs = epochs[int(n_windows/2)-1:-int(n_windows/2)]

for key in all_fitnesses:
    # Calculate mean and statistics
    f_mean = np.mean(all_fitnesses[key],axis=0)
    f_std = np.std(all_fitnesses[key],axis=0)

    
    f_mean = np.convolve(f_mean, np.ones(n_windows) / n_windows, mode='valid')
    f_std = np.convolve(f_std, np.ones(n_windows) / n_windows, mode='valid')

    ax.semilogy(epochs, f_mean, label=r'$T={0}\ \mathrm{{s}}$'.format(key))
    ax.fill_between(epochs, f_mean - f_std, f_mean+f_std,  alpha=0.5)

key = 't_max'
ax.set_xlabel('Epoch',fontsize=12)
ax.set_ylabel(r'$\bar{f}$', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim([0, 5000])
ax.grid(True, which="both")
ax.set_box_aspect(1)
ax.set_yticks([0.1, 0.2, 0.5, 1.0])
ax.set_yticklabels([0.1, 0.2, 0.5, 1.0])

ax.set_ylim([0.06, 1.5])

plt.savefig('./figures/convergence/{0}.png'.format(key), dpi=300, bbox_inches='tight')

plt.show()