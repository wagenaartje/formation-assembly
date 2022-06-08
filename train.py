## Imports
import math
import numpy as np
import time
import json

from pymoo.model.problem import Problem
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

## Custom imports
from evaluation import evaluate_population

## Import configuration
from config import config


# Initialize output streams
timestamp = int(time.time())
fitness_file = open('./results/{0}_f.dat'.format(timestamp),mode='wb+')
genome_file = open('./results/{0}_g.dat'.format(timestamp),mode='wb+')
config_file = open('./results/{0}_c.json'.format(timestamp), 'w')

json.dump(config, config_file, sort_keys=True, indent=4)
config_file.close()

## Define the problem for pymoo
class FormationProblem (Problem):
    def __init__(self):

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -100 * np.ones(config['n_param'])
        xu = +100 * np.ones(config['n_param'])

        super().__init__(n_var=config['n_param'], n_obj=1, n_constr=0, xl=xl, xu=xu)


    def _evaluate(self, x, out, *args, **kwargs):
        fitnesses  = evaluate_population(config, x)
        out["F"] = fitnesses

        # Save the best genome and fitness
        best_fitness = np.min(fitnesses)
        best_fitness.tofile(fitness_file)

        best_genome  = x[[np.argmin(fitnesses)],:]
        best_genome.tofile(genome_file)
    
        # Log some results
        print(np.min(fitnesses), np.mean(fitnesses))


problem = FormationProblem();


algorithm = CMAES(
    x0=np.random.random(config['n_param'])* 1 - 0.5,
    sigma=0.2,
    popsize=config['n_genomes']
)

## Run the EA
termination = SingleObjectiveDefaultTermination(
    x_tol=0,
    cv_tol=0,
    f_tol=0.001,
    nth_gen=math.inf,
    n_last=math.inf,
    n_max_gen=10000,
    n_max_evals=math.inf
)

res = minimize(problem, algorithm, verbose=True, termination=termination)

