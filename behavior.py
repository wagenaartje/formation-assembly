import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import json
from evaluation import single_evaluate

run = './results/base/1654698799/'

fitnesses = np.fromfile(run + 'fitnesses.dat')
genomes = np.fromfile(run + 'genomes.dat').reshape((fitnesses.shape[0], -1))
with open(run + 'config.json') as f:
    config = json.load(f)

config['dt'] = 0.001

# Add related parameters
config['n_inputs'] = config['n_agents']*4 # Input neurons
config['n_outputs'] = 2                   # Output neurons

# Total number of parameters
config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']


# Select the last genome
best_genome = genomes[[-1],:]

# Simulate once
f1, f2, formation, position_history, collided = single_evaluate(config, best_genome, analysis=True)


#print(bcs)

print('Fitness:', f1, f2, collided)

# Plot the results
fig, axes = plt.subplots(2,2)

axes[0,0].scatter(formation[0,:,0], formation[0,:,1],label='Target',s=60,c='white',edgecolor='black')
axes[0,0].set_aspect(1)

#axes[0].scatter(initial_position[0,:,0], initial_position[0,:,1],label='Start')

com = np.mean(position_history,axis=2)


com = np.mean(position_history,axis=2)

#axes[0,1].scatter(com[0,0,0], com[0,0,1])
#axes[0,1].scatter(com[-1,0,0], com[-1,0,1])


for i in range(config['n_agents']):
    axes[0,1].plot(position_history[:,0,i,0], position_history[:,0,i,1],'k:')
    axes[0,1].text(position_history[-1,0,i,0]+0.05, position_history[-1,0,i,1]+0.07, f'{i}')

axes[0,1].scatter(position_history[0,0,:,0], position_history[0,0,:,1],c='white',label='Start',edgecolor='black', linewidth=1, s=60,hatch='///////',zorder=10)
axes[0,1].scatter(position_history[-1,0,:,0], position_history[-1,0,:,1],c='white',label='End',edgecolor='black', linewidth=1, s=60,zorder=10)

xlim = axes[0,1].get_xlim()
ylim = axes[0,1].get_ylim()
xsize = np.diff(xlim)
xcent = np.mean(xlim)
ysize = np.diff(ylim)
ycent = np.mean(ylim)

if xsize > ysize:
    axes[0,1].set_ylim([ycent - xsize/2, ycent + xsize/2])
    biggest = xsize
else:
    axes[0,1].set_xlim([xcent - ysize/2, xcent + ysize/2])
    biggest = ysize

xlim = axes[0,0].get_xlim()
ylim = axes[0,0].get_ylim()
xcent = np.mean(xlim)
ycent = np.mean(ylim)

axes[0,0].set_xlim([xcent - biggest/2, xcent + biggest/2])
axes[0,0].set_ylim([ycent - biggest/2, ycent + biggest/2])

axes[0,1].set_aspect(1)






axes[0,1].legend()



''' Velocity & acceleration '''
velocity = np.linalg.norm(np.diff(position_history,n=1,axis=0),axis=3)
time = config['dt']* np.arange(velocity.shape[0])

print(velocity.shape)

for i in range(config['n_agents']):
    axes[1,0].plot(time,velocity[:,0,i] / config['dt'], label=str(i))

axes[1,0].set_box_aspect(1)
axes[1,0].legend()


acceleration = np.linalg.norm(np.diff(position_history,n=2,axis=0),axis=3) / config['dt']**2
time = config['dt']* np.arange(acceleration.shape[0])

for i in range(config['n_agents']):
    axes[1,1].plot(time,acceleration[:,0,i],label=str(i))

axes[1,1].legend()

# NOTE! Maybe create a heatmap out of this?
# acceleration_components = np.diff(position_history,n=2,axis=0) 
#for i in range(config['n_agents']):
#     axes[1,1].plot(acceleration_components[:,0,i,0],  acceleration_components[:,0,i,1])


axes[1,1].set_box_aspect(1)


plt.show()
