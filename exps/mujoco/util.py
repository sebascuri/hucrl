"""Utility files to run experiments."""
import yaml


def parse_config_file(file_dir):
    """Parse configuration file."""
    with open(file_dir, "r") as file:
        args = yaml.safe_load(file)
    return args
