n_size = 4        # Size of the world (number of cells)
n_agents = 2     # Number of agents in the world
n_steps = 100     # Number of movement steps to simulate
n_genomes = 100
n_evals = 1

n_hidden = 64
n_inputs = (n_agents-1)*4
n_outputs = 2

#             weights + biases + weights + biases 

n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs

p_c = 0.5 # Crossover probability
p_m = 0.025 # Mutation probability