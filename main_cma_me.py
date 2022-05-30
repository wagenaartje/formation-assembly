import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from settings import *
from evaluation import evaluate_population

from ribs.optimizers import Optimizer
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.visualize import grid_archive_heatmap

# x: total acceleration  (this does not mean we get smooth paths per se!! )
# y: velocity when reaching target
# objective: mean distance to target

archive = GridArchive(
    [100, 100],  # 50 bins in each dimension.
    [(0, 1.0), (0,1)],  # (-1, 1) for x-pos and (-3, 0) for y-vel.
)

# Load best genomes and their fitnesses
genomes = np.fromfile('./output/genome.dat')
genomes = np.reshape(genomes, (-1, n_param))
fitnesses = np.fromfile('./output/fitness.dat')
print(np.min(fitnesses))

# Select the genome with the best fitness
initial_model = genomes[np.argmin(fitnesses)]


emitters = [
    ImprovementEmitter(
        archive,
        initial_model.flatten(),
        1.0,  # Initial step size.
        batch_size=30,
    ) for _ in range(5)  # Create 5 separate emitters.
]



optimizer = Optimizer(archive, emitters)

start_time = time.time()
total_itrs = 50000

plt.figure(figsize=(8, 6))


for itr in tqdm(range(1, total_itrs + 1)):
    # Request models from the optimizer.
    sols = optimizer.ask()

    # Evaluate the models and record the objectives and BCs.
    objs, bcs = evaluate_population(sols)

    # Send the results back to the optimizer.
    optimizer.tell(objs, bcs)

    # Logging.
    print(itr, len(archive), archive.stats.obj_max)

    # Test something here
    if itr % 10 == 0:
        plt.clf()
        grid_archive_heatmap(archive, vmin=-1, vmax=0)
        plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
        plt.ylabel("Time")
        plt.xlabel("Dist")
                
        plt.savefig('archive.png')


# With 32 neurons, -0.11 as best fitness
# With 64 neurons, not much difference