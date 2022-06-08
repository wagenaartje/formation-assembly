import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

t_maxs = [2, 5, 10, 20]

for i in range(5):
    for t_max in t_maxs:
        base_config['t_max'] = t_max

        train('/tmax/', base_config)