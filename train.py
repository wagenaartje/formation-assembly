## Imports
import math
import numpy as np
import time
import json5, json
import os

from pymoo.model.problem import Problem
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.util.display import Display
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

## Custom imports
from evaluation import evaluate_population

## Define the problem for pymoo
class FormationProblem (Problem):
    def __init__(self, config: dict, fitness_file, genome_file):
        self.fitness_file = fitness_file
        self.genome_file = genome_file
        self.config = config

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -100 * np.ones(config['n_param'])
        xu = +100 * np.ones(config['n_param'])

        super().__init__(n_var=config['n_param'], n_obj=1, n_constr=0, xl=xl, xu=xu)


    def _evaluate(self, x, out, *args, **kwargs):
        fitnesses  = evaluate_population(self.config, x)
        out["F"] = fitnesses

        # Save the best genome and fitness
        best_fitness = np.min(fitnesses)
        best_fitness.tofile(self.fitness_file)

        best_genome  = x[[np.argmin(fitnesses)],:]
        best_genome.tofile(self.genome_file)


class MyDisplay(Display):
    def __init__ (self):
        self.start = time.time()

        super().__init__()

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append('f_opt', algorithm.opt.get('F')[0][0])
        self.output.append("f_min", np.min(algorithm.pop.get("F")))
        self.output.append("f_mean", np.mean(algorithm.pop.get("F")))
        self.output.append("time", time.time() - self.start)

        self.start = time.time()


def train (subfolder: str, config: dict) -> None:
    # Initialize output streams
    timestamp = int(time.time())
    os.makedirs('./results/{0}/'.format(subfolder), exist_ok=True)
    os.makedirs('./results/{0}/{1}/'.format(subfolder, timestamp), exist_ok=True)
    fitness_file = open('./results/{0}/{1}/fitnesses.dat'.format(subfolder, timestamp),mode='wb+')
    genome_file = open('./results/{0}/{1}/genomes.dat'.format(subfolder, timestamp),mode='wb+')
    config_file = open('./results/{0}/{1}/config.json'.format(subfolder, timestamp), 'w')

    json.dump(config, config_file, sort_keys=True, indent=4)
    config_file.close()

    # Add related parameters
    config['n_inputs'] = config['n_agents']*4 # Input neurons
    config['n_outputs'] = 2                   # Output neurons

    # Total number of parameters
    config['n_param'] = config['n_hidden']*config['n_inputs'] + config['n_hidden'] + config['n_hidden'] * config['n_outputs'] + config['n_outputs']

    problem = FormationProblem(config, fitness_file, genome_file);

    algorithm = CMAES(
        x0=np.random.random(config['n_param'])* 1 - 0.5,
        sigma=0.2,
        popsize=config['n_genomes']
    )

    ## Run the EA
    termination = SingleObjectiveDefaultTermination(
        x_tol=0,
        cv_tol=0,
        f_tol=0,
        nth_gen=math.inf,
        n_last=math.inf,
        n_max_gen=config['n_gen'],
        n_max_evals=math.inf
    )

    minimize(problem, algorithm, verbose=True, termination=termination, display=MyDisplay())

if __name__ == '__main__':
    # Import configuration
    with open('base_config.json') as f:
        config = json5.load(f)

    train('', config)

