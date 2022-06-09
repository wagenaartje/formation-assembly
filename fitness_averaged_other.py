import numpy as np
import json
import os
import matplotlib.pyplot as plt

folder = '/amax/'
key = 'a_max'

fig,ax = plt.subplots(1)

epochs = np.arange(5001)

all_fitnesses = {}

subfolders = os.listdir('./results/' + folder)
for subfolder in subfolders:
    fitnesses = -np.fromfile('./results/' + folder +  '/' + subfolder + '/fitnesses.dat')
    with open('./results/' + folder +  '/' + subfolder + '/config.json') as f:
        config = json.load(f)


    if fitnesses.shape[0] == epochs.shape[0]:
        if config[key] not in all_fitnesses:
            all_fitnesses[config[key]] = []
        all_fitnesses[config[key]].append(fitnesses)

print(len(all_fitnesses))

n_windows = 50
epochs = epochs[int(n_windows/2)-1:-int(n_windows/2)]

for key in all_fitnesses:
    # Calculate mean and statistics
    f_mean = np.mean(all_fitnesses[key],axis=0)
    f_std = np.std(all_fitnesses[key],axis=0)

    
    f_mean = np.convolve(f_mean, np.ones(n_windows) / n_windows, mode='valid')
    f_std = np.convolve(f_std, np.ones(n_windows) / n_windows, mode='valid')

    ax.plot(epochs, f_mean, label=key)
    ax.fill_between(epochs, f_mean - f_std, f_mean+f_std,  alpha=0.5)

ax.set_xlabel('Epoch')
ax.set_ylabel('Fitness')
ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(top=0)
ax.grid(True, which="both")
ax.set_box_aspect(1)

plt.show()