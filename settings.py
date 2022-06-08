''' Simulation settings '''
n_agents = 3     # Number of agents
n_genomes = 100  # Number of genomes
n_evals = 10     # Number of evaluations for fitness

min_dist = 0.3   # Minimum distance to satisfy [m]
max_acc = 2      # Maximum acceleration [m/s^2]
max_vel = 1      # Maximum velocity [m/s]

dt = 0.01
t_max = 10

# NOTE! Add timestep here.

''' Neural network settings '''
n_hidden = 16               # Hidden neurons
n_inputs = (n_agents)*4    # Input neurons
n_outputs = 2              # Output neurons

# Total number of parameters
n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs

def to_str () -> str:
    ''' Converts the settings to a string sequence '''
    return 'agents={0}hidden={1}evals={2}t_max={3}pop={4}dist={5}dt={6}'.format(
        n_agents,
        n_hidden,
        n_evals,
        t_max,
        n_genomes,
        min_dist,
        dt
    )


def from_file (file_name: str) -> dict:
    ''' Takes a file name and returns settings in dictionary form '''
    result = {}

    result['n_agents'] = int(file_name.split('=')[1].split('h')[0])
    result['n_hidden'] = int(file_name.split('=')[2].split('e')[0])
    result['n_evals'] = int(file_name.split('=')[3].split('t')[0])
    result['t_max'] = int(file_name.split('=')[4].split('p')[0])
    result['n_genomes'] = int(file_name.split('=')[5].split('d')[0])
    result['min_dist'] = float(file_name.split('=')[6].split('d')[0])
    result['dt'] = float(file_name.split('=')[7][:-4])

    return result