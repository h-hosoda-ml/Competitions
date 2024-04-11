import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import DIRS

_all_datas = [
    os.path.join(DIRS["DATA"], "train.csv"),
    os.path.join(DIRS["DATA"], "test.csv"),
]


class DataLoader:
    @staticmethod
    def load_data(paths: list = _all_datas) -> list[pd.DataFrame]:
        dataframes = []
        for path in paths:
            dataframes.append(pd.read_csv(path))

        return dataframes
