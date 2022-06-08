import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

hs = [8, 16, 32, 64]

for i in range(5):
    for h in hs:
        base_config['n_hidden'] = h

        train('/hidden/', base_config)