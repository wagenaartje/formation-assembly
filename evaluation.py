import numpy as np
from settings import *
import itertools

def population_action (population,inputs):
    w1 = population[:,:n_hidden*n_inputs].reshape((population.shape[0],inputs.shape[2], n_hidden))
    b1 = population[:,n_hidden*n_inputs:n_hidden*(n_inputs+1)].reshape((population.shape[0], 1, n_hidden))
    w2 = population[:,n_hidden*(n_inputs+1):n_hidden*(n_inputs+1+n_outputs)].reshape((population.shape[0],n_hidden, n_outputs))
    b2 = population[:,-n_outputs:].reshape((population.shape[0], 1, n_outputs))


    #print(inputs.shape, w1.shape, np.einsum('ijk,ikp->ijp', inputs,w1).shape)
    #print(inputs[0].dot(w1[0]))
    #print(np.einsum('ijk,ikp->ijp', inputs,w1)[0])

    z1 = np.einsum('ijk,ikp->ijp', inputs,w1) + b1
    a1 = np.tanh(z1)

    z2 = np.einsum('ijk,ikp->ijp', a1,w2) + b2
    a2 = np.tanh(z2)

    return a2


        
def generate_formation ():
    formation = np.zeros((n_agents, 2))

    direction_vector = np.random.rand(2) *2 - 1
    direction_vector /= np.linalg.norm(direction_vector)

    formation[1,:] = direction_vector

    direction_vector = np.random.rand(2) *2 - 1
    direction_vector /= np.linalg.norm(direction_vector)

    formation[2,:] = formation[1,:] + direction_vector

    formation = np.reshape(formation, (1,3,2))
    formation -= formation[:,[0],:]

    return formation


permutations = list(itertools.permutations(range(n_agents),n_agents))


def single_evaluate(population, save=False):
    # They start in a 2x2 area
    initial_position = np.random.rand(1,n_agents,2) * 4 - 2
    positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)
    formation = generate_formation()

    formation_input = np.repeat(formation[:,[1,2],:],population.shape[0],axis=0)
    formation_input = np.reshape(formation_input, (population.shape[0], 1,4))

    if save:
        np.save('data/formation.npy', formation)
        np.save('data/init_pos.npy', initial_position)
        position_history = np.zeros((n_steps, population.shape[0], n_agents, 2))

    for i in range(n_steps):
        # Gather inputs for 1st agent
        inputs_0 = positions[:,[1,2],:] - positions[:,[0],:]
        inputs_0 = np.reshape(inputs_0, (population.shape[0],1,4))
        inputs_0 = np.concatenate((inputs_0, formation_input), axis=2)

        # Gather inputs for 2nd agent
        inputs_1 = positions[:,[0,2],:] - positions[:,[1],:]
        inputs_1 = np.reshape(inputs_1, (population.shape[0],1,4))
        inputs_1 = np.concatenate((inputs_1, formation_input), axis=2)

        # Gather inputs for 3nd agent
        inputs_2 = positions[:,[0,1],:] - positions[:,[2],:]
        inputs_2 = np.reshape(inputs_2, (population.shape[0],1,4))
        inputs_2 = np.concatenate((inputs_2, formation_input), axis=2)

        # Concenate to 3 samples per genome
        inputs = np.concatenate((inputs_0,inputs_1,inputs_2),axis=1)

        # Get action
        velocities = population_action(population, inputs)

        positions += velocities * 0.05

        if save:
            position_history[i] = positions

    if save:
        np.save('data/pos_history.npy', position_history)

    # Now at the end, compare to formation
    positions -= np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation -= np.reshape(np.mean(formation,axis=1),(1,1,2))

    # Now, we have to go over all possible combinations and take the minimum
    fitnesses = np.ones(population.shape[0]) * np.inf;

    
    
    for order in permutations:
        rel_locations = positions[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation,axis=2),axis=1)

        fitnesses = np.where(rel_dist_diff < fitnesses, rel_dist_diff, fitnesses);

    return fitnesses

def evaluate_population (population):
    total_fitnesses = np.zeros(population.shape[0])

    for j in range(n_evals):
        fitnesses = single_evaluate(population)
        total_fitnesses += fitnesses

    total_fitnesses /= n_evals

    return total_fitnesses