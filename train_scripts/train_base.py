import sys
sys.path.append('.')

from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

base_config['dt'] = 0.01
# NOTE! We should also do 10000 generations.

for i in range(10):
    train('/base/', base_config)
