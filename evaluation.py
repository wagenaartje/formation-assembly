import numpy as np
import itertools

def population_action (config: dict, population: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    ''' Calculates the action of each genome using a single-layer neural-network '''

    # Gather the weights and biases
    w1 = population[:,:config['n_hidden']*config['n_inputs']].reshape((population.shape[0],config['n_inputs'], config['n_hidden']))
    b1 = population[:,config['n_hidden']*config['n_inputs']:config['n_hidden']*(config['n_inputs']+1)].reshape((population.shape[0], 1, config['n_hidden']))
    w2 = population[:,config['n_hidden']*(config['n_inputs']+1):config['n_hidden']*(config['n_inputs']+1+config['n_outputs'])].reshape((population.shape[0],config['n_hidden'], config['n_outputs']))
    b2 = population[:,-config['n_outputs']:].reshape((population.shape[0], 1, config['n_outputs']))

    # Activations of first layer
    z1 = np.einsum('ijk,ikp->ijp', inputs,w1) + b1
    a1 = np.tanh(z1)

    # Activations of second layer
    z2 = np.einsum('ijk,ikp->ijp', a1,w2) + b2
    a2 = np.tanh(z2)

    return a2

def gen_points_min_dist (config: dict) -> np.ndarray:
    points = np.zeros((1,config['n_agents'],2))

    i = 0
    while True:
        new_point = np.random.rand(2) * 4 - 2

        good = True
        for j in range(0, i):
            if np.linalg.norm(new_point - points[0,j]) < config['d_min']:
                good = False
                break
        
        if good:
            points[0,i] = new_point
            i += 1
        
        if i == config['n_agents']: break

    return points

def single_evaluate(config: dict, population: np.ndarray, save: bool = False) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population on a single evaluation'''
    permutations = list(itertools.permutations(range(config['n_agents']),config['n_agents']))

    loops = int(config['t_max']/config['dt'])

    # Initialize random initial positions and formations
    initial_position = gen_points_min_dist(config)

    positions = np.repeat(initial_position.copy(), population.shape[0],axis=0)
    
    formation = gen_points_min_dist(config)
    formation_input = np.repeat(formation.copy(),population.shape[0],axis=0)
    formation_input -= np.mean(formation_input,axis=1,keepdims=True)

    # If save=true, log the position history
    if save: position_history = np.zeros((loops, population.shape[0], config['n_agents'], 2))

    velocity = np.zeros((population.shape[0],config['n_agents'],2)) 
    collided = np.ones(population.shape[0])

    # Now, we have to go over all possible combinations and take the minimum
    for i in range(loops):
        inputs = np.zeros((population.shape[0],0,config['n_inputs']))
        
        # NOTE! We must keep this here, since we subtract formation_input and positions later.
        positions_centered = positions - np.mean(positions,axis=1,keepdims=True)
        

        if save: position_history[i] = positions.copy()

        for j in range(config['n_agents']):
            # Gather inputs for 1st agent
            agents = np.delete(np.arange(config['n_agents']),j)
            np.random.shuffle(agents)

            formation_select = np.arange(config['n_agents'])
            np.random.shuffle(formation_select)

            inputs_0 = positions_centered[:,agents,:] - positions_centered[:,[j],:]

            distances = np.min(np.linalg.norm(inputs_0,axis=2),axis=1)
            collided = np.where(distances < config[ 'd_min'], np.zeros(population.shape[0]), collided)

            inputs_0 = np.reshape(inputs_0, (population.shape[0],1,(config['n_agents']-1)*2))

            formation_enter = formation_input[:,formation_select,:] - positions_centered[:,[j],:]
            formation_enter = np.reshape(formation_enter, (population.shape[0], 1,config['n_agents']*2))
            inputs_0 = np.concatenate((velocity[:,[j],:], inputs_0, formation_enter), axis=2)


            # Concatenate to 3 samples per genome
            inputs = np.concatenate((inputs,inputs_0),axis=1)
        
        # Get action
        acceleration = population_action(config, population, inputs) * config['a_max']

        acceleration_magnitude = np.linalg.norm(acceleration,axis=2,keepdims=True)
        acceleration = np.where(acceleration_magnitude < config['a_max'], acceleration, acceleration / acceleration_magnitude * config[ 'a_max'])

        velocity += acceleration * config['dt']


        # Clip the velocity
        velocity_magnitude = np.linalg.norm(velocity,axis=2,keepdims=True)
        velocity = np.where(velocity_magnitude < config['v_max'], velocity, velocity / velocity_magnitude * config['v_max'])

        # Update the position (if not collided!)
        collided_full = np.reshape(collided, (population.shape[0],1,1))
        positions += collided_full * velocity * config['dt']

    # If save=True, save initial position, formation, and path
    if save:
        np.save('./tmp/initial_position.npy', initial_position)
        np.save('./tmp/formation.npy', formation)
        np.save('./tmp/position_history.npy', position_history)

    # Now at the end, compare to formation
    positions_c = positions.copy() - np.reshape(np.mean(positions,axis=1),(population.shape[0],1,2))
    formation_c = formation - np.reshape(np.mean(formation,axis=1),(1,1,2))

    f1 = np.ones(population.shape[0]) * np.inf
    for order in permutations:
        rel_locations = positions_c[:,list(order),:]
        rel_dist_diff = np.mean(np.linalg.norm(rel_locations - formation_c,axis=2),axis=1)

        f1 = np.where(rel_dist_diff < f1, rel_dist_diff, f1) 

    # Boundary conditions
    f2 = np.mean(np.linalg.norm(velocity,axis=2),axis=1)

    return f1 + f2

def evaluate_population (config: dict, population: np.ndarray) -> np.ndarray:
    ''' Calculates the fitness of each genome in the population, averaged over n_evals '''
    fitnesses = np.zeros(population.shape[0])

    for _ in range(config['n_evals']):
        fitnesses += single_evaluate(config, population)

    fitnesses /= config['n_evals']


    return fitnesses