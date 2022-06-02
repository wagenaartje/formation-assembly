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

def gen_points_min_dist (min_dist):
    points = np.zeros((1,n_agents,2))

    i = 0
    while True:
        new_point = np.random.rand(2) * 4 - 2

        good = True
        for j in range(0, i):
            if np.linalg.norm(new_point - points[0,j]) < min_dist:
                good = False
                break
        
        if good:
            points[0,i] = new_point
            i += 1
        
        if i == n_agents: break

    return points

def single_evaluate(population: np.ndarray, loops: int, lt_fitness: bool = False, save: bool = False) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population on a single evaluation'''

    # Initialize random initial positions and formations
    initial_position = gen_points_min_dist(0.2)

    positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)
    
    formation = gen_points_min_dist(0.2)
    formation_input = np.repeat(formation.copy(),population.shape[0],axis=0)
    formation_input -= np.mean(formation_input,axis=1,keepdims=True)

    # Now at the end, compare to formation
    positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    old_best_diff = np.ones((population.shape[0])) * np.inf
    for order in permutations:
        rel_locations = positions_c[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)

        old_best_diff = np.where(rel_dist_diff < old_best_diff, rel_dist_diff, old_best_diff) 

    # If save=true, log the position history
    if save: position_history = np.zeros((loops, population.shape[0], n_agents, 2))

    bc2 = np.zeros((population.shape[0], 1))
    velocity = np.zeros((population.shape[0],n_agents,2)) # NOTE to self: shouldn't we start with a random velocity perhaps?
    collided = np.ones(population.shape[0])

    # Now, we have to go over all possible combinations and take the minimum
    for i in range(loops):
        inputs = np.zeros((population.shape[0],0,n_inputs))
        
        # NOTE! We must keep this here, since we subtract formation_input and positions later.
        positions_centered = positions - np.mean(positions,axis=1,keepdims=True)
        

        if save: position_history[i] = positions.copy()

        for j in range(n_agents):
            # Gather inputs for 1st agent
            agents = np.delete(np.arange(n_agents),j)
            np.random.shuffle(agents)

            formation_select = np.arange(n_agents)
            np.random.shuffle(formation_select)

            inputs_0 = positions_centered[:,agents,:] - positions_centered[:,[j],:]

            distances = np.min(np.linalg.norm(inputs_0,axis=2),axis=1)

            collided = np.where(distances < 0.2, np.zeros(population.shape[0]), collided)

            inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(n_agents-1)*2))

            formation_enter = formation_input[:,formation_select,:] - positions_centered[:,[j],:]
            formation_enter = np.reshape(formation_enter, (population.shape[0], 1,n_agents*2))
            inputs_0 = np.concatenate((velocity[:,[j],:], inputs_0, formation_enter), axis=2)


            # Concatenate to 3 samples per genome
            inputs = np.concatenate((inputs,inputs_0),axis=1)
        
        # Get action
        old_velocity = velocity.copy()

        acceleration = population_action(population, inputs)
        velocity += acceleration * 0.05

        collided_full = np.reshape(collided, (population.shape[0],1,1))

        velocity = collided_full * np.clip(velocity,-1,1)
        positions += velocity * 0.05
        
        # NOTE! Don't use "acceleration" here, because it is not true acceleration.
        bc2 += np.mean(np.linalg.norm(velocity - old_velocity,axis=2),axis=1,keepdims=True)

        


    # Now at the end, compare to formation
    positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    best_diff = np.ones((population.shape[0])) * np.inf
    for order in permutations:
        rel_locations = positions_c[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)

        best_diff = np.where(rel_dist_diff < best_diff, rel_dist_diff, best_diff) 

    if not lt_fitness:
        fitness = -(old_best_diff - best_diff)
    else:
        fitness = best_diff
        
    # If save=True, save initial position, formation, and path
    if save:
        np.save('./tmp/initial_position.npy', initial_position)
        np.save('./tmp/formation.npy', formation)
        np.save('./tmp/position_history.npy', position_history)


    bc1 = np.mean(np.linalg.norm(velocity,axis=2),axis=1, keepdims=True)

    # and direction!
    bc2 = np.mean(np.arctan2(velocity[:,:,1], velocity[:,:,0]),axis=1,keepdims=True)

    bcs = np.concatenate((bc1, bc2),axis=1)

    return fitness, bcs

def evaluate_population (population: np.ndarray, loops: int, lt_fitness: bool = False) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population, averaged over n_evals '''
    fitnesses = np.zeros(population.shape[0])
    bcs = np.zeros((population.shape[0], 2))

    for _ in range(n_evals):
        eval_fitness, eval_bcs = single_evaluate(population, loops, lt_fitness)
        fitnesses += eval_fitness
        bcs += eval_bcs

    fitnesses /= n_evals
    bcs /= n_evals

    # print(np.mean(bcs,axis=0), np.max(bcs,axis=0))

    return fitnesses, bcs