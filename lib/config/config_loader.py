import yaml

class ConfigLoader:
    def __init__(self, config_path="./config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load the configuration from a YAML file."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config