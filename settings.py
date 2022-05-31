''' Simulation settings '''
n_agents = 3     # Number of agents
n_steps = 100    # Number of steps per evaluation
n_genomes = 100  # Number of genomes
n_evals = 10     # Number of evaluations for fitness

p_c = 0.5        # Crossover probability
p_m = 0.05       # Mutation probability

''' Neural network settings '''
n_hidden = 8               # Hidden neurons
n_inputs = (n_agents)*4 -2  # Input neurons
n_outputs = 2               # Output neurons

# Total number of parameters
n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs

def to_str () -> str:
    ''' Converts the settings to a string sequence '''
    return 'agents={0}hidden={1}evals={2}steps={3}pop={4}pc={5}pm={6}'.format(
        n_agents,
        n_hidden,
        n_evals,
        n_steps,
        n_genomes,
        p_c,
        p_m
    )


def from_file (file_name: str) -> dict:
    ''' Takes a file name and returns settings in dictionary form '''
    result = {}

    result['n_agents'] = int(file_name.split('=')[1].split('h')[0])
    result['n_hidden'] = int(file_name.split('=')[2].split('e')[0])
    result['n_evals'] = int(file_name.split('=')[3].split('s')[0])
    result['n_steps'] = int(file_name.split('=')[4].split('p')[0])
    result['n_genomes'] = int(file_name.split('=')[5].split('p')[0])
    result['p_c'] = float(file_name.split('=')[6].split('p')[0])
    result['p_m'] = float(file_name.split('=')[7][:-4])

    return result