import numpy as np
from settings import *
import itertools

def population_action (population,inputs):
    w1 = population[:,:n_hidden*n_inputs].reshape((population.shape[0],n_inputs, n_hidden))
    b1 = population[:,n_hidden*n_inputs:n_hidden*(n_inputs+1)].reshape((population.shape[0], 1, n_hidden))
    w2 = population[:,n_hidden*(n_inputs+1):n_hidden*(n_inputs+1+n_outputs)].reshape((population.shape[0],n_hidden, n_outputs))
    b2 = population[:,-n_outputs:].reshape((population.shape[0], 1, n_outputs))

    # print(population[0])
    # print(w1[0])
    # print(b1[0])
    # print(w2[0])
    # print(b2[0])

    #print(inputs.shape, w1.shape, np.einsum('ijk,ikp->ijp', inputs,w1).shape)
    #print(inputs[0].dot(w1[0]))
    #print(np.einsum('ijk,ikp->ijp', inputs,w1)[0])

    z1 = np.einsum('ijk,ikp->ijp', inputs,w1) + b1

    #print(w1.shape,inputs.shape,b1.shape)
    #print(np.tanh(np.dot(np.tanh(inputs[0][0].dot(w1[0])+b1[0]), w2[0]) + b2[0]))
    #print(z2[0][0])
    a1 = np.tanh(z1)

    z2 = np.einsum('ijk,ikp->ijp', a1,w2) + b2
    a2 = np.tanh(z2)


    return a2

def fixed_action (population, inputs):
    permutations = list(itertools.permutations(range(n_agents),n_agents))
    formations = np.concatenate((np.reshape(inputs[:,:,(n_agents-1)*2:], (population.shape[0], n_agents, 2, 2)), np.zeros((population.shape[0], n_agents, 1, 2))),axis=2)
    formations -= np.reshape(np.mean(formations,axis=2), (population.shape[0], n_agents, 1, 2))
    #print(formations.shape)

    best = np.inf
    best_order = None

    relative_positions = np.reshape(inputs[:,:,:(n_agents-1)*2], (population.shape[0], n_agents, 2, 2))

    best_diff = np.ones((population.shape[0],3)) * np.inf
    best_order_index = np.zeros((population.shape[0],n_agents))
    for i in range(len(permutations)):
        order = permutations[i]
        # permute formation, and see for which relative distances is most similar.
        formation_copy = formations[:,:,list(order),:].copy()
        formation_copy -= formation_copy[:,:,[0],:]

        rel_dist_diff = np.mean(np.linalg.norm(relative_positions - formation_copy[:,:,1:],axis=3),axis=2)

        best_order_index = np.where(rel_dist_diff < best_diff, np.ones((population.shape[0],n_agents)) * i, best_order_index)
        best_diff = np.where(rel_dist_diff < best_diff, rel_dist_diff, best_diff)

    
    best_orders = np.asarray(permutations)[best_order_index.astype('int')]

    formation_copy = np.zeros((population.shape[0], n_agents, n_agents, 2))

    for i in range(population.shape[0]):
        for j in range(3):
            formation_copy[i,j] = formations[i,j,best_orders[i,j]]
    formation_copy -= formation_copy[:,:,[0],:]

    #print(i, best_order, best, relative_positions, formation_copy[1:])

    rel_dist_diff = np.sum(relative_positions - formation_copy[:,:,1:],axis=2)

    rel_dist_diff = np.sign(rel_dist_diff)

    return rel_dist_diff


        
def generate_formation ():
    formation = np.zeros((n_agents, 2))

    for j in range(1,n_agents):
        direction_vector = np.random.rand(2) *2 - 1
        direction_vector /= np.linalg.norm(direction_vector)

        formation[j,:] = formation[j-1,:] + direction_vector

    formation = np.reshape(formation, (1,n_agents,2))
    formation -= formation[:,[0],:]

    return formation


permutations = list(itertools.permutations(range(n_agents),n_agents))


def single_evaluate(population, save=False):
    # They start in a 2x2 area
    initial_position = np.random.rand(1,n_agents,2) * 4 - 2
    positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)
    formation = generate_formation()

    formation_input = np.repeat(formation[:,np.arange(1,n_agents),:],population.shape[0],axis=0)
    formation_input = np.reshape(formation_input, (population.shape[0], 1,(n_agents-1)*2))

    if save:
        np.save('data/formation.npy', formation)
        np.save('data/init_pos.npy', initial_position)
        position_history = np.zeros((n_steps, population.shape[0], n_agents, 2))

    for i in range(n_steps):
        inputs = np.zeros((population.shape[0],0,n_inputs))
        for j in range(n_agents):
            # Gather inputs for 1st agent
            agents = np.delete(np.arange(n_agents),j)
            np.random.shuffle(agents)

            inputs_0 = positions[:,agents,:] - positions[:,[j],:]
            inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(n_agents-1)*2))
            inputs_0 = np.concatenate((inputs_0, formation_input), axis=2)

            # Concenate to 3 samples per genome
            inputs = np.concatenate((inputs,inputs_0),axis=1)
            
        # Get action
        velocities = population_action(population, inputs)

        positions += velocities * 0.05

        if save:
            position_history[i] = positions

    if save:
        np.save('data/pos_history.npy', position_history)

    # Now at the end, compare to formation
    positions -= np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    # Now, we have to go over all possible combinations and take the minimum
    fitnesses = np.ones(population.shape[0]) * np.inf;

    
    
    for order in permutations:
        rel_locations = positions[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)

        fitnesses = np.where(rel_dist_diff < fitnesses, rel_dist_diff, fitnesses);

    return fitnesses

def evaluate_population (population):
    total_fitnesses = np.zeros(population.shape[0])

    for j in range(n_evals):
        fitnesses = single_evaluate(population)
        total_fitnesses += fitnesses

    total_fitnesses /= n_evals

    return total_fitnesses