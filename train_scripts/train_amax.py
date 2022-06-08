import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

a_maxes = [0.5, 1, 2, 4]

for a_max in a_maxes:
    base_config['a_max'] = a_max

    for i in range(10):
        train('/amax/', base_config)