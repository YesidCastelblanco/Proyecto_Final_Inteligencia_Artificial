import yaml
from pathlib import Path

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

config = load_config()