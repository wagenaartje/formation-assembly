import numpy as np
import json
import os
import matplotlib.pyplot as plt
import urllib

folder = '/base/'
keys = ['n_evals']


fig,ax = plt.subplots(1)
#for folder in os.listdir('./results/'):
subfolders = os.listdir('./results/' + folder)
for subfolder in subfolders:
    fitnesses = -np.fromfile('./results/' + folder +  '/' + subfolder + '/fitnesses.dat')
    with open('./results/' + folder +  '/' + subfolder + '/config.json') as f:
        config = json.load(f)
    
    relevant_config = {}
    for key in keys:
        relevant_config[key] = config[key]
    label = urllib.parse.urlencode(relevant_config)
    
    # NOTE to self: in the future, calculate std from unconvolved data. but then convolve std. 
    epochs = np.arange(fitnesses.shape[0])

    n_windows = 50 # must be even
    fitnesses = np.convolve(fitnesses, np.ones(n_windows) / n_windows, mode='valid')


    if n_windows != 1:
        ax.plot(epochs[int(n_windows/2)-1:-int(n_windows/2)],fitnesses, label= folder + label)
    else:
        ax.plot(epochs,fitnesses, label=folder + label)

ax.set_xlabel('Epoch')
ax.set_ylabel('Fitness')
ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(top=0)
ax.grid()

plt.show()