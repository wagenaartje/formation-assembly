import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

dts = [0.005, 0.01, 0.05, 0.10]

for i in range(5):
    for dt in dts:
        base_config['dt'] = dt

        train('/dt/', base_config)