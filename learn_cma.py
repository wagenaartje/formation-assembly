## Imports
import math
import numpy as np

from pymoo.model.problem import Problem
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

from settings import *

## Custom imports
#from simulate import simulate
from evaluation import evaluate_population


# Initialize output streams
settings_str = to_str()
fitness_file = open('./runs/f_{0}.dat'.format(settings_str),mode='wb+')
genome_file = open('./runs/g_{0}.dat'.format(settings_str),mode='wb+')

## Define the problem for pymoo
class MyProblem(Problem):
    def __init__(self):

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -100 * np.ones(n_param)
        xu = +100 * np.ones(n_param)

        super().__init__(n_var=n_param, n_obj=1, n_constr=0, xl=xl, xu=xu)


    def _evaluate(self, x, out, *args, **kwargs):
        fitnesses, _ = evaluate_population(x, n_steps, lt_fitness=True)
        out["F"] = fitnesses

        # Save the best genome and fitness
        best_fitness = np.min(fitnesses)
        best_fitness.tofile(fitness_file)

        best_genome  = x[[np.argmin(fitnesses)],:]
        best_genome.tofile(genome_file)
    
        # Log some results
        print(np.min(fitnesses), np.mean(fitnesses))


problem = MyProblem();


algorithm = CMAES(
    x0=np.random.random(n_param)* 1 - 0.5,
    sigma=0.2,
    popsize=n_genomes
)

## Run the EA
termination = SingleObjectiveDefaultTermination(
    x_tol=0,
    cv_tol=0,
    f_tol=0.01,
    nth_gen=math.inf,
    n_last=math.inf,
    n_max_gen=50000,
    n_max_evals=math.inf
)

res = minimize(problem,
               algorithm,
               verbose=True,
               termination = termination)

