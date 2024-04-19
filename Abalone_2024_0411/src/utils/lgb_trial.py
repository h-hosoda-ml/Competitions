import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import optuna
from optuna.trial import Trial
import lightgbm as lgb
import optuna.integration.lightgbm as optlgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import *


# クロスバリデーションによるスコア計算
# validationsの中のものと違い、trainデータのみで行う
def model_cv(X: pd.DataFrame, y: pd.DataFrame, params: dict):
    kf = KFold(n_splits=5, shuffle=True, random_state=PARAMS["seed"])
    scores = []  # スコアを格納

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
        dtrain, dval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_val, y_val)

        # 訓練
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(200),
            ],
        )
        # 予測
        preds = gbm.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, preds))

        # 記録
        scores.append(score)

    return np.mean(scores)


# Optunaによるパラメータ探索
def regular_tuning(trial: Trial, X, y):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": PARAMS["seed"],
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "max_leaves": trial.suggest_int("num_leaves", 20, 40),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.5, 1.0
        ),  # subsampleと同義
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.5, 1.0
        ),  # colsample_bytreeと同義
    }

    score = model_cv(X, y, params)

    return score


def new_tuning(X: pd.DataFrame, y: pd.DataFrame, params: dict) -> lgb.Booster:
    kf = KFold(n_splits=5, shuffle=True, random_state=PARAMS["seed"])
    tr_idx, va_idx = list(kf.split(X, y))[0]

    X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
    dtrain, dval = optlgb.Dataset(X_train, y_train), optlgb.Dataset(X_val, y_val)

    # 訓練
    gbm = optlgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(500),
        ],
    )

    return gbm
