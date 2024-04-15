import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import DIRS, CFG

_all_datas = [
    os.path.join(DIRS["DATA"], "train.csv"),
    os.path.join(DIRS["DATA"], "test.csv"),
]


class DataLoader:
    @staticmethod
    def load_data(paths: list = _all_datas, id_col: str = CFG.ID) -> list[pd.DataFrame]:
        dataframes = []
        for path in paths:
            dataframe = pd.read_csv(path, index_col=id_col)
            dataframes.append(dataframe)

        return dataframes
