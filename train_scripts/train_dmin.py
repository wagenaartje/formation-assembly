import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

d_mins = [0, 0.1, 0.2, 0.3]

for i in range(5):
    for d_min in d_mins:
        base_config['d_min'] = d_min
        
        train('/dmin/', base_config)