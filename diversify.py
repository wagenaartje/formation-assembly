import time
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from evaluation import evaluate_population

from ribs.optimizers import Optimizer
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.visualize import grid_archive_heatmap

# NOTE: I think we should use long-term fitness here. But we have to test if algorithm works when we use that. (it should)

archive = GridArchive(
    [50, 50],  # 50 bins in each dimension.
    [(0, np.sqrt(2)), (0,1)],  # for velocities and average acceleration
)

# Load best genomes and their fitnesses
genomes = np.fromfile('./runs/g_' + to_str() + '.dat')
genomes = np.reshape(genomes, (-1, n_param))
#fitnesses = np.fromfile('./output/fitness.dat')

# Select the genome with the best fitness
initial_model = genomes[-1]

#initial_model = np.zeros(n_param)

# NOTE! We could also do emitters with different initial_models / sigmas!
sigmas = [0.01, 0.05, 0.1,0.2,0.5,1]
emitters = [
    ImprovementEmitter(
        archive,
        initial_model.flatten(),
        sigmas[i],  # Initial step size.
        batch_size=30,
    ) for i in range(6)  # Create 5 separate emitters.
]



optimizer = Optimizer(archive, emitters)

start_time = time.time()
total_itrs = 500000

plt.figure(figsize=(8, 6))


for itr in range(1, total_itrs + 1):
    # Request models from the optimizer.
    sols = optimizer.ask()

    # Evaluate the models and record the objectives and BCs.
    objs, bcs = evaluate_population(sols, n_steps, lt_fitness=True)
    objs = -objs # check this??

    # Send the results back to the optimizer.
    optimizer.tell(objs, bcs)

    # Logging.
    print(itr, len(archive), archive.stats.obj_max)

    # Test something here
    if itr % 10 == 0:
        plt.clf()
        grid_archive_heatmap(archive, vmin=-1, vmax=0)
        plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
        plt.xlabel("Mean velocity")
        plt.ylabel("Arrival time")
                
        plt.savefig('archive.png')


# With 32 neurons, -0.11 as best fitness
# With 64 neurons, not much difference