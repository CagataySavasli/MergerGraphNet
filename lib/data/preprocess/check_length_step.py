from preprocess_step import PreprocessStep
import pandas as pd

class CheckLengthStep(PreprocessStep):
    def __init__(self, column: str, minimum_token_limit: int):
        self.column = column
        self.minimum_token_limit = minimum_token_limit
        
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['word_count'] = df[self.column].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= self.minimum_token_limit]
        df.drop(columns=['word_count'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df