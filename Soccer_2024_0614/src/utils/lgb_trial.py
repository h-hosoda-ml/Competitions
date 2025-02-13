import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import optuna.integration.lightgbm as optlgb

import optuna
from optuna.trial import Trial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf.config import *
from utils.validations import time_cross_val


def time_cross_val_tuning(trial: Trial, X: pd.DataFrame, y: pd.Series, config: CFG):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": 42,
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

    scores = time_cross_val(X, y, params, config)

    prev_scores = scores[:-1]

    score_avg = np.mean([scores[-1], np.mean(prev_scores)])

    return score_avg


def new_tuning(
    X: pd.DataFrame, y: pd.DataFrame, params: dict, config: CFG
) -> lgb.Booster:
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    tr_idx, va_idx = list(kf.split(X, y))[0]

    X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]

    dtrain, dval = (
        optlgb.Dataset(
            data=X_train,
            label=y_train,
            feature_name=config.FEATURES,
            categorical_feature=config.CAT_FEATURES,
        ),
        optlgb.Dataset(
            data=X_val,
            label=y_val,
            feature_name=config.FEATURES,
            categorical_feature=config.CAT_FEATURES,
        ),
    )

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


def new_tuning_along_time(
    X: pd.DataFrame, y: pd.DataFrame, df_test: pd.DataFrame, params: dict, config: CFG
) -> lgb.Booster:
    if not config.PERIOD:
        config.PERIOD = [2010, 2014, 2017]

    models = []
    test_preds = np.zeros((len(df_test)))
    val_preds = np.zeros((len(X)))
    val_scores = []

    for va_period in config.PERIOD:
        is_tr = X["year"] < va_period
        is_va = X["year"] == va_period

        X_train, X_val = X[is_tr], X[is_va]
        y_train, y_val = y[is_tr], y[is_va]

        dtrain, dval = (
            optlgb.Dataset(
                data=X_train,
                label=y_train,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
            optlgb.Dataset(
                data=X_val,
                label=y_val,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
        )

        # 訓練
        LGB = optlgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(500),
            ],
            verbosity=0,
        )
        y_pred_tr = LGB.predict(X_train)
        y_pred_val = LGB.predict(X_val)

        train_rmsle = np.sqrt(mean_squared_error(y_train, y_pred_tr))
        val_rmsle = np.sqrt(mean_squared_error(y_val, y_pred_val))
        print(f"Train RMSLE: {train_rmsle:.4f}  Val RMSLE: {val_rmsle:.4f}")
        test_preds += LGB.predict(df_test) / len(config.PERIOD)
        val_preds[is_va] = LGB.predict(X_val)
        val_scores.append(val_rmsle)
        models.append(LGB)
        print("-" * 50)

    return models, val_scores, val_preds, test_preds
