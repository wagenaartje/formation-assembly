''' Simulation settings '''
config = {}

config['n_agents'] = 3    # Number of agents
config['n_genomes'] = 100 # Number of genomes
config['n_evals'] = 10    # Number of evaluations for fitness

config['d_min'] = 0.3     # Minimum distance to satisfy [m]
config['a_max'] = 2       # Maximum acceleration [m/s^2]
config['v_max'] = 1       # Maximum velocity [m/s]

config['dt'] = 0.01       # Timestep [s]
config['t_max'] = 5      # Simulation time [s]


''' Neural network settings '''
config['n_hidden'] = 16                   # Hidden neurons
config['n_inputs'] = config['n_agents']*4 # Input neurons
config['n_outputs'] = 2                   # Output neurons

# Total number of parameters
config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']