import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

v_maxes = [0.5, 1, 2, 4]

for i in range(5):
    for v_max in v_maxes:
        base_config['v_max'] = v_max

        train('/vmax/', base_config)