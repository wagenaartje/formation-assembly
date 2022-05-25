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
    a1 = z1 * (z1 > 0)

    z2 = np.einsum('ijk,ikp->ijp', a1,w2) + b2
    a2 = np.tanh(z2)


    return a2

def fixed_action (population, inputs):
    permutations = list(itertools.permutations(range(n_agents),n_agents))
    formations = np.concatenate((np.reshape(inputs[:,:,(n_agents-1)*2:], (population.shape[0], n_agents, n_agents-1, 2)), np.zeros((population.shape[0], n_agents, 1, 2))),axis=2)
    formations -= np.reshape(np.mean(formations,axis=2), (population.shape[0], n_agents, 1, 2))
    #print(formations.shape)

    best = np.inf
    best_order = None

    relative_positions = np.reshape(inputs[:,:,:(n_agents-1)*2], (population.shape[0], n_agents, n_agents-1, 2))

    best_diff = np.ones((population.shape[0],n_agents)) * np.inf
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
        for j in range(n_agents):
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



archive = [[1,1]]

best_genome = None

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

    # Now, we have to go over all possible combinations and take the minimum
    behavior = np.ones((population.shape[0],2)) * np.inf;
    best_diff = np.ones((population.shape[0],1)) * np.inf
    time_diff = np.ones((population.shape[0],1)) * np.inf

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
        #if i == n_steps-1 and population.shape[0] > 1:print(velocities[0,:,:])
        positions += velocities * 0.05

        if save:
            position_history[i] = positions


        # Now at the end, compare to formation
        positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
        formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

        for order in permutations:
            rel_locations = positions_c[:,list(order),:]
            rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)
            rel_dist_diff = np.reshape(rel_dist_diff, (population.shape[0],1))

            time_diff = np.where(rel_dist_diff < best_diff, np.ones((population.shape[0],1)) * i/n_steps, time_diff)
            best_diff = np.where(rel_dist_diff < best_diff, rel_dist_diff, best_diff)
    if save:
        np.save('data/pos_history.npy', position_history)


    
    
    behavior = np.concatenate((best_diff, time_diff),axis=1)
    behavior = np.clip(behavior,0,1)


    return behavior, best_diff[:,0]

best_fitness = np.inf
def evaluate_population (population):
    global archive, best_genome, best_fitness

    ''' Get behavior '''
    average_behavior = np.zeros((population.shape[0],2))
    fitnesses = np.zeros(population.shape[0])


    for j in range(n_evals):
        behavior, fitness  = single_evaluate(population)
        average_behavior += behavior
        fitnesses += fitness

    average_behavior /= n_evals
    fitnesses /= n_evals

    novelty = np.zeros(population.shape[0])

    for i in range(population.shape[0]):
        # But this is weird. If we have multiple novel but identical individuals...
        all_others = np.concatenate((archive, average_behavior[:i], average_behavior[i+1:]))

        distances = np.abs(np.linalg.norm(all_others - average_behavior[i],axis=1))
        distances = np.sort(distances)

        k = 1
        minimum_distances = distances[:k]
        novelty[i] = - np.mean(minimum_distances)

    if np.min(fitnesses) < best_fitness:
        best_fitness = np.min(fitnesses)
        best_genome = population[[np.argmin(fitnesses)],:]

    single_evaluate(best_genome,save=True)

    archive = archive + list(average_behavior[novelty < -0.01 ])

    print(best_fitness, np.mean(average_behavior,axis=0), len(archive))

    np.save('archive.npy', np.asarray(archive))


    return novelty