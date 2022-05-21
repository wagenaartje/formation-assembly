import numpy as np
from settings import *
import itertools
from evaluation import generate_formation

# Formation


# Define action
def action (relative_positions, formation, i):
    permutations = list(itertools.permutations(range(n_agents),n_agents))

    formation = np.concatenate(([[0,0]], formation), axis=0)
    formation -= np.mean(formation,axis=0)

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

    #print(i, best_order, best, relative_positions, formation_copy[1:])

    rel_dist_diff = np.sum(relative_positions - formation_copy[1:],axis=0)


    rel_dist_diff = np.sign(rel_dist_diff)

    return rel_dist_diff

if __name__ == '__main__':
    total_fitness = 0
    for k in range(100):
        formation = generate_formation()[0]
        formation -= formation[[0],:]
        # Initial position
        position = np.random.rand(n_agents,2) * 4 - 2


        for i in range(500):
            # Gather inputs for 1st agent
            inputs_0 = position[[1,2],:] - position[[0],:]
            velocity_0 = action(inputs_0, formation[[1,2],:], 0)

            # Gather inputs for 2nd agent

            inputs_1 = position[[0,2],:] - position[[1],:]
            velocity_1 = action(inputs_1, formation[[1,2],:], 1)

            # Gather inputs for 3nd agent
            inputs_2 = position[[0,1],:] - position[[2],:]
            velocity_2 = action(inputs_2, formation[[1,2],:],2)

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

        print(fitness)
        total_fitness += fitness

    print(total_fitness/100)

    import matplotlib.pyplot as plt

    plt.scatter(position[:,0], position[:,1])
    plt.scatter(formation[:,0], formation[:,1])
    plt.show()