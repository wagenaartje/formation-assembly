import numpy as np
from settings import *
import itertools

def genome_action (genome, inputs):
    w1 = genome[:n_hidden*4].reshape((inputs.shape[1], n_hidden))
    b1 = genome[n_hidden*4:n_hidden*5]
    w2 = genome[n_hidden*5:n_hidden*7].reshape((n_hidden, 2))
    b2 = genome[-2:]

    z1 = np.dot(inputs, w1) + b1
    a1 = np.tanh(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = np.tanh(z2)

    return a2


def population_action (population,inputs):
    w1 = population[:,:n_hidden*4].reshape((population.shape[0],inputs.shape[2], n_hidden))
    b1 = population[:,n_hidden*4:n_hidden*5].reshape((population.shape[0], 1, n_hidden))
    w2 = population[:,n_hidden*5:n_hidden*7].reshape((population.shape[0],n_hidden, 2))
    b2 = population[:,-2:].reshape((population.shape[0], 1, 2))

    z1 = np.einsum('ijk,izp->ijp', inputs,w1) + b1
    a1 = np.tanh(z1)

    z2 = np.einsum('ijk,izp->ijp', a1,w2) + b2
    a2 = np.tanh(z2)


    return a2


        
def generate_formation ():
    formation = np.zeros((n_agents, 2))

    direction_vector = np.random.rand(2) *2 - 1
    direction_vector /= np.linalg.norm(direction_vector)

    formation[1,:] = direction_vector

formation = np.asarray([[0,0], [1/np.sqrt(2), 1/np.sqrt(2)], [2/np.sqrt(2),0]])
formation = np.reshape(formation, (1,3,2))
formation -= np.reshape(np.mean(formation,axis=1),(1,1,2))

permutations = list(itertools.permutations(range(n_agents),n_agents))


def single_evaluate(population, save=False):
    # They start in a 2x2 area
    initial_position = np.random.rand(1,n_agents,2) * 4 - 2
    positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)

    positions_save = np.zeros((n_steps, population.shape[0], n_agents, 2))

    for i in range(n_steps):
        positions_save[i] = positions
        # Gather inputs for 1st agent
        inputs_0 = positions[:,[1,2],:] - positions[:,[0],:]
        inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(n_agents-1)*2))

        # Gather inputs for 2nd agent
        inputs_1 = positions[:,[0,2],:] - positions[:,[1],:]
        inputs_1 = np.reshape(inputs_1, (population.shape[0],1,(n_agents-1)*2))

        # Gather inputs for 3nd agent
        inputs_2 = positions[:,[0,1],:] - positions[:,[2],:]
        inputs_2 = np.reshape(inputs_2, (population.shape[0],1,(n_agents-1)*2))

        # Concenate to 3 samples per genome
        inputs = np.concatenate((inputs_0,inputs_1,inputs_2),axis=1)

        # Get action
        velocities = population_action(population, inputs)

        positions += velocities * 0.05

    # Now at the end, compare to formation
    positions -= np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))

    # Now, we have to go over all possible combinations and take the minimum
    fitnesses = np.ones(population.shape[0]) * np.inf;
    
    for order in permutations:
        rel_locations = positions[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation,axis=2),axis=1)

        fitnesses = np.where(rel_dist_diff < fitnesses, rel_dist_diff, fitnesses);

    np.save('positions_save.npy', positions_save)

def evaluate_population (population):
    total_fitnesses = np.zeros(population.shape[0])
    old_positions = np.zeros((n_evals, population.shape[0], n_agents, 2))
    for j in range(n_evals):
        # They start in a 2x2 area
        initial_position = np.random.rand(1,n_agents,2) * 4 - 2
        positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)

        for i in range(n_steps):
            # Gather inputs for 1st agent
            inputs_0 = positions[:,[1,2],:] - positions[:,[0],:]
            inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(n_agents-1)*2))

            # Gather inputs for 2nd agent
            inputs_1 = positions[:,[0,2],:] - positions[:,[1],:]
            inputs_1 = np.reshape(inputs_1, (population.shape[0],1,(n_agents-1)*2))

            # Gather inputs for 3nd agent
            inputs_2 = positions[:,[0,1],:] - positions[:,[2],:]
            inputs_2 = np.reshape(inputs_2, (population.shape[0],1,(n_agents-1)*2))

            # Concenate to 3 samples per genome
            inputs = np.concatenate((inputs_0,inputs_1,inputs_2),axis=1)

            # Get action
            velocities = population_action(population, inputs)

            positions += velocities * 0.05

        # Now at the end, compare to formation
        positions -= np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))

        # Now, we have to go over all possible combinations and take the minimum
        fitnesses = np.ones(population.shape[0]) * np.inf;
        
        for order in permutations:
            rel_locations = positions[:,list(order),:]
            rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation,axis=2),axis=1)

            fitnesses = np.where(rel_dist_diff < fitnesses, rel_dist_diff, fitnesses);

        total_fitnesses += fitnesses
        old_positions[j] = positions

    best_positions = np.zeros((n_evals, 3, 2))

    for i in range(n_evals):
        best_positions[i] = old_positions[i,np.argmin(total_fitnesses)]

    np.save('best_position.npy', best_positions)

    total_fitnesses /= n_evals

    return total_fitnesses