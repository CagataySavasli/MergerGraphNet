import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    A utility class for loading and managing configuration data from a YAML file.
    """

    def __init__(self, config_path: str = "./config.yaml") -> None:
        """
        Initialize the ConfigLoader with a given configuration file path.

        Args:
            config_path (str): The path to the YAML configuration file.
                               Defaults to './config.yaml'.
        """
        self._config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the configuration from the specified YAML file.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If the YAML file does not exist at the given path.
            yaml.YAMLError: If an error occurs while parsing the YAML file.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self._config_path}")

        with self._config_path.open("r", encoding="utf-8") as file:
            try:
                config_data = yaml.safe_load(file)
                # Return an empty dictionary if the file is empty.
                return config_data if config_data else {}
            except yaml.YAMLError as err:
                raise yaml.YAMLError(f"Error parsing YAML file: {err}")

    @property
    def config(self) -> Dict[str, Any]:
        """
        Retrieve the loaded configuration data.

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary.
        """
        return self._config