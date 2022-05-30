import numpy as np
from settings import *
import itertools

def population_action (population: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    ''' Calculates the action of each genome using a single-layer neural-network '''

    # Gather the weights and biases
    w1 = population[:,:n_hidden*n_inputs].reshape((population.shape[0],n_inputs, n_hidden))
    b1 = population[:,n_hidden*n_inputs:n_hidden*(n_inputs+1)].reshape((population.shape[0], 1, n_hidden))
    w2 = population[:,n_hidden*(n_inputs+1):n_hidden*(n_inputs+1+n_outputs)].reshape((population.shape[0],n_hidden, n_outputs))
    b2 = population[:,-n_outputs:].reshape((population.shape[0], 1, n_outputs))

    # Activations of first layer
    z1 = np.einsum('ijk,ikp->ijp', inputs,w1) + b1
    a1 = np.tanh(z1)

    # Activations of second layer
    z2 = np.einsum('ijk,ikp->ijp', a1,w2) + b2
    a2 = np.tanh(z2)

    return a2

permutations = list(itertools.permutations(range(n_agents),n_agents))

def single_evaluate(population: np.ndarray, loops: int) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population on a single evaluation'''

    # Initialize random initial positions and formations
    initial_position = np.random.rand(1,n_agents,2) * 4 - 2
    positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)
    
    formation = np.random.rand(1,n_agents, 2) * 4 - 2
    formation_input = np.repeat(formation.copy(),population.shape[0],axis=0)

    # Now at the end, compare to formation
    positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    old_best_diff = np.ones((population.shape[0])) * np.inf
    for order in permutations:
        rel_locations = positions_c[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)

        old_best_diff = np.where(rel_dist_diff < old_best_diff, rel_dist_diff, old_best_diff) 


    # Now, we have to go over all possible combinations and take the minimum
    for i in range(loops):
        inputs = np.zeros((population.shape[0],0,n_inputs))
        
        # NOTE! We must keep this here, since we subtract formation_input and positions later.
        positions -= np.mean(positions,axis=1,keepdims=True)
        formation_input -= np.mean(formation_input,axis=1,keepdims=True)

        for j in range(n_agents):
            # Gather inputs for 1st agent
            agents = np.delete(np.arange(n_agents),j)
            np.random.shuffle(agents)

            formation_select = np.arange(n_agents)
            np.random.shuffle(formation_select)

            inputs_0 = positions[:,agents,:] - positions[:,[j],:]
            inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(n_agents-1)*2))

            formation_enter = formation_input[:,formation_select,:] - positions[:,[j],:]
            formation_enter = np.reshape(formation_enter, (population.shape[0], 1,n_agents*2))
            inputs_0 = np.concatenate((inputs_0, formation_enter), axis=2)


            # Concatenate to 3 samples per genome
            inputs = np.concatenate((inputs,inputs_0),axis=1)
        
        # Get action
        velocities = population_action(population, inputs)
        positions += velocities * 0.05


    # Now at the end, compare to formation
    positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    best_diff = np.ones((population.shape[0])) * np.inf
    for order in permutations:
        rel_locations = positions_c[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)

        best_diff = np.where(rel_dist_diff < best_diff, rel_dist_diff, best_diff) 



    fitness = -(old_best_diff - best_diff)


    return fitness

def evaluate_population (population: np.ndarray) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population, averaged over n_evals '''
    fitnesses = np.zeros(population.shape[0])

    for _ in range(n_evals):
        fitnesses += single_evaluate(population, n_steps)

    fitnesses /= n_evals


    return fitnesses