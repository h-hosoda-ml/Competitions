import os
import sys
from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from conf import CFG


# データの読み込み
class DataProcesser:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.X_train = train.drop([CFG.TARGET], axis=1)
        self.y_train = train[CFG.TARGET]

        self.test = test

    def preprocess(self):
        self.X_train["Sex"] = self.X_train["Sex"].astype("category")
        self.test["Sex"] = self.test["Sex"].astype("category")

        self.__null_checker()

        print(f"Train shape: {self.X_train.shape}   test shape: {self.test.shape}")

        display(self.X_train)

        return {"X_train": self.X_train, "y_train": self.y_train}, self.test

    def __null_checker(self):
        if not self.X_train.isnull().any().any():
            print("train dataの中に欠損値はありませんでした。")

        if not self.test.isnull().any().any():
            print("test dataの中に欠損値はありませんでした。")


# データエンジニアリング
class DataEnginner:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.X_train = train
        self.X_test = test

    def execute(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # カテゴリ変数をlabel encoding
        for col in CFG.CAT_FEATURES:
            le = LabelEncoder()
            le.fit(self.X_train[col])

            self.X_train[col] = le.transform(self.X_train[col])
            self.X_test[col] = le.transform(self.X_test[col])

        # 数値データをログに変換
        log_features = []
        for col in CFG.NUMERIC_FEATURES:
            self.X_train[f"log_{col}"] = np.log1p(self.X_train[col])
            self.X_test[f"log_{col}"] = np.log1p(self.X_test[col])
            log_features.append(f"log_{col}")

        # 数値データの標準化を行う
        scaler = StandardScaler()
        scaler.fit(self.X_train[CFG.NUMERIC_FEATURES])

        self.X_train[CFG.NUMERIC_FEATURES] = scaler.transform(
            self.X_train[CFG.NUMERIC_FEATURES]
        )
        self.X_test[CFG.NUMERIC_FEATURES] = scaler.transform(
            self.X_test[CFG.NUMERIC_FEATURES]
        )

        return self.X_train, self.X_test
