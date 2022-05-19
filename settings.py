n_size = 4        # Size of the world (number of cells)
n_agents = 3     # Number of agents in the world
n_steps = 200     # Number of movement steps to simulate
n_genomes = 100
n_evals = 10

n_hidden = 16

#             weights + biases + weights + biases 
n_param = n_hidden*(4) + n_hidden + n_hidden * 2 + 2
print(n_param)

p_c = 0.5 # Crossover probability
p_m = 0.3 # Mutation probability