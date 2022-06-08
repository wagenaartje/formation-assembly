import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

evals = [1, 5, 10, 20]

for e in evals:
    base_config['n_evals'] = e

    for i in range(1):
        train('/evals/', base_config)