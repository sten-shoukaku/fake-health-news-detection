import os
import pandas as pd
import re
import typing


class CoAID:
    """CoAID dataset helper"""

    def __init__(self, root_path: str):
        dir_list = [
            os.path.join(root_path, d)
            for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d)) and d != '.git'
        ]
        file_list = self.__get_filenames(*dir_list)
        df = pd.concat([self.__open_csv(f) for f in file_list])
        self.df = df

    def get_data(self) -> pd.DataFrame:
        """Get dataset as Pandas DataFrame"""
        return self.df

    def __get_filenames(self, *dir_path: str) -> typing.List[str]:
        filenames = []
        for path in dir_path:
            filenames += [
                os.path.join(path, f)
                for f in os.listdir(path)
                if 'tweets' not in f.lower()
            ]
        return filenames

    def __open_csv(
        self,
        file_path: str,
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df['target'] = 1 if 'real' in file_path.lower() else 0
        df = df.rename(columns={'title': 'input'})
        df = self.__preprocess(df)
        return df[['input', 'target']]

    def __preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['input'] = df['input'].apply(lambda x: re.sub(r'^"|"$', '', x))
        df['input'] = df['input'].apply(lambda x: x.lower())
        return df
