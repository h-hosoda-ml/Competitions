import os
import sys
from typing import Optional

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf.config import DIRS, CFG


# データ群
__TRAIN_DF = pd.read_csv(os.path.join(DIRS["DATA"], "train.csv"))
__TEST_DF = pd.read_csv(os.path.join(DIRS["DATA"], "test.csv"))
__VENUE_DF = pd.read_csv(os.path.join(DIRS["DATA"], "venue_information.csv"))
__DETAIL_DF = pd.read_csv(os.path.join(DIRS["DATA"], "match_reports.csv"))


# データの呼び出しを行う関数
def load_all_datas() -> list[pd.DataFrame]:
    train_df = _merge_subdatas(__TRAIN_DF).copy()
    test_df = _merge_subdatas(__TEST_DF).copy()
    return [train_df, test_df]


# マージを行う関数
def _merge_subdatas(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df, __DETAIL_DF, left_on="id", right_on="id", how="inner")
    df = pd.merge(df, __VENUE_DF, left_on="venue", right_on="venue", how="inner")
    return df
