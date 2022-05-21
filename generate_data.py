import numpy as np
from evaluation import generate_formation
from manual_control import action
import itertools

n_agents = 3
n_runs_data = 2000
n_steps = 10
n_inputs = 8
n_outputs = 2

''' Generate data '''
x_data = np.zeros((n_runs_data, n_steps, n_agents, n_inputs))
y_data = np.zeros((n_runs_data, n_steps, n_agents, n_outputs))


total_fitness = 0
for k in range(n_runs_data):
    formation = generate_formation()[0]
    formation -= formation[[0],:]

    # Initial position
    position = np.random.rand(n_agents,2) * 4 - 2

    for i in range(n_steps):
        # Gather inputs for 1st agent
        inputs_0 = position[[1,2],:] - position[[0],:]
        velocity_0 = action(inputs_0, formation[[1,2],:], 0)
        inputs_0_flat = np.concatenate((inputs_0.flatten(), formation[[1,2],:].flatten()), axis=0)
        x_data[k,i,0,:] = inputs_0_flat
        y_data[k,i,0,:] = velocity_0
    

        # Gather inputs for 2nd agent
        inputs_1 = position[[0,2],:] - position[[1],:]
        velocity_1 = action(inputs_1, formation[[1,2],:], 1)
        inputs_1_flat = np.concatenate((inputs_1.flatten(), formation[[1,2],:].flatten()), axis=0)
        x_data[k,i,1,:] = inputs_1_flat
        y_data[k,i,1,:] = velocity_1

        # Gather inputs for 3nd agent
        inputs_2 = position[[0,1],:] - position[[2],:]
        velocity_2 = action(inputs_2, formation[[1,2],:],2)
        inputs_2_flat = np.concatenate((inputs_2.flatten(), formation[[1,2],:].flatten()), axis=0)
        x_data[k,i,2,:] = inputs_2_flat
        y_data[k,i,2,:] = velocity_2

        # Concatenate to 3 samples per genome
        velocities = np.vstack((velocity_0, velocity_1, velocity_2))

        # Get action
        position += velocities * 0.05



    # Determine and print error
    position -= np.reshape(np.mean(position,axis=0),(1,2))
    formation -= np.mean(formation,axis=0)
    # Now, we have to go over all possible combinations and take the minimum
    fitness = np.inf

    permutations = list(itertools.permutations(range(n_agents),n_agents))
    for order in permutations:
        rel_locations = position[list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation,axis=1),axis=0)

        if rel_dist_diff < fitness:
            fitness = rel_dist_diff

    total_fitness += fitness


x_data = np.reshape(x_data, (n_runs_data * n_steps * n_agents, n_inputs))
y_data = np.reshape(y_data, (n_runs_data * n_steps * n_agents, n_outputs))

np.save('data/x_data.npy', x_data)
np.save('data/y_data.npy', y_data)


print(total_fitness/n_runs_data)