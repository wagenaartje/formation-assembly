''' Simulation settings '''
n_agents = 3     # Number of agents
n_steps = 100     # Number of steps per evaluation
n_genomes = 100  # Number of genomes
n_evals = 10    # Number of evaluations for fitness
n_steps_lt = 200 # Long term number of steps per evaluation

p_c = 0.5        # Crossover probability
p_m = 0.05       # Mutation probability

''' Neural network settings '''
n_hidden = 32               # Hidden neurons
n_inputs = (n_agents)*4 -2  # Input neurons
n_outputs = 2               # Output neurons

# Total number of parameters
n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs

