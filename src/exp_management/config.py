"""
Abstract config class for ML-run definitions
"""
from abc import ABC, abstractmethod
from typing import List

import configargparse

class Config(ABC):
    """
    Abstract config class
    """

    def __init__(self, config_paths: List[str]) -> None:
        self.config_paths = config_paths

    @abstractmethod
    def parse_config(self,
                     testmode: bool = True,
                     verbose: bool = True
                     ) -> configargparse.Namespace:
        """Creates a args namespace object from a config file

        Args:
            verbose (bool): Set True to print args

        Returns:
            configargparse.Namespace: args
        """
