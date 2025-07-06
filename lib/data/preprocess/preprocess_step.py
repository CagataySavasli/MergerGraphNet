from abc import ABC, abstractmethod
import pandas as pd

class PreprocessStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass