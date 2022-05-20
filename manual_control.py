import numpy as np
from settings import *
import itertools

# Formation
formation = np.asarray([[0,0], [1/np.sqrt(2), 1/np.sqrt(2)], [2/np.sqrt(2),0]])
formation = np.reshape(formation, (1,3,2))
formation -= np.reshape(np.mean(formation,axis=1),(1,1,2))
formation = formation[0]

# Define action
def action (relative_positions):
    permutations = list(itertools.permutations(range(n_agents),n_agents))

    best = np.inf
    best_order = None
    for order in permutations:
        # permute formation, and see for which relative distances is most similar.
        formation_copy = formation[list(order),:].copy()
        formation_copy -= formation_copy[[0],:]

        rel_dist_diff = np.mean(np.linalg.norm(relative_positions - formation_copy[1:],axis=1),axis=0)

        if rel_dist_diff < best:
            best = rel_dist_diff
            best_order = order

    formation_copy = formation[list(best_order),:].copy()
    formation_copy -= formation_copy[[0],:]

    print(best_order)

    rel_dist_diff = np.sum(relative_positions - formation_copy[1:],axis=0)


    rel_dist_diff = np.sign(rel_dist_diff)

    return rel_dist_diff

# Initial position
position = np.random.rand(n_agents,2) * 4 - 2


for i in range(1000):
    # Gather inputs for 1st agent
    inputs_0 = position[[1,2],:] - position[[0],:]
    velocity_0 = action(inputs_0)

    # Gather inputs for 2nd agent

    inputs_1 = position[[0,2],:] - position[[1],:]
    velocity_1 = action(inputs_1)

    # Gather inputs for 3nd agent
    inputs_2 = position[[0,1],:] - position[[2],:]
    velocity_2 = action(inputs_1)

    # Concatenate to 3 samples per genome
    velocities = np.vstack((velocity_0, velocity_1, velocity_2))

    # Get action
    position += velocities * 0.05


# Determine and print error
position -= np.reshape(np.mean(position,axis=0),(1,2))

# Now, we have to go over all possible combinations and take the minimum
fitness = np.inf

permutations = list(itertools.permutations(range(n_agents),n_agents))
for order in permutations:
    rel_locations = position[list(order),:]
    rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation,axis=1),axis=0)

    if rel_dist_diff < fitness:
        fitness = rel_dist_diff

print(rel_dist_diff)
