from train import train
import json5

# Load the base config
with open('base_config.json') as f:
    base_config = json5.load(f)

train(base_config)
