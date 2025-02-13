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
import optuna.integration.lightgbm as optlgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf.config import *

_val_period_list = [2009, 2012, 2015, 2017]


def cross_val_train(X: pd.DataFrame, y: pd.DataFrame, df_test, params, config: CFG):
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    test_preds = np.zeros((len(df_test)))
    val_preds = np.zeros((len(X)))
    val_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_train = X.iloc[tr_idx]
        y_train = y[tr_idx]
        X_val = X.iloc[va_idx]
        y_val = y[va_idx]

        dtrain, dval = (
            lgb.Dataset(
                data=X_train,
                label=y_train,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
            lgb.Dataset(
                data=X_val,
                label=y_val,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
        )

        LGB = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=350,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(500),
            ],
        )

        y_pred_tr = LGB.predict(X_train)
        y_pred_val = LGB.predict(X_val)

        train_rmsle = np.sqrt(mean_squared_error(y_train, y_pred_tr))
        val_rmsle = np.sqrt(mean_squared_error(y_val, y_pred_val))
        print(
            f"Fold: {fold}   Train RMSLE: {train_rmsle:.4f}  Val RMSLE: {val_rmsle:.4f}"
        )
        test_preds += LGB.predict(df_test) / kf.get_n_splits()
        val_preds[va_idx] = LGB.predict(X_val)
        val_scores.append(val_rmsle)
        print("-" * 50)

    return val_scores, val_preds, test_preds, LGB


def cross_val_opt_train(X: pd.DataFrame, y: pd.DataFrame, df_test, params, config: CFG):
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    test_preds = np.zeros((len(df_test)))
    val_preds = np.zeros((len(X)))
    val_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_train = X.iloc[tr_idx]
        y_train = y[tr_idx]
        X_val = X.iloc[va_idx]
        y_val = y[va_idx]

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

        LGB = optlgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=75),
                lgb.log_evaluation(100),
            ],
        )

        y_pred_tr = LGB.predict(X_train)
        y_pred_val = LGB.predict(X_val)

        train_rmsle = np.sqrt(mean_squared_error(y_train, y_pred_tr))
        val_rmsle = np.sqrt(mean_squared_error(y_val, y_pred_val))
        print(
            f"Fold: {fold}   Train RMSLE: {train_rmsle:.4f}  Val RMSLE: {val_rmsle:.4f}"
        )
        test_preds += LGB.predict(df_test) / kf.get_n_splits()
        val_preds[va_idx] = LGB.predict(X_val)
        val_scores.append(val_rmsle)
        print("-" * 50)

    return val_scores, val_preds, test_preds, LGB


def walk_forward_time_val(X: pd.DataFrame, y: pd.DataFrame, params: dict, config: CFG):
    rmse_scores = []

    # 年ごとにデータの分割
    for i, val_period in enumerate(_val_period_list):
        if i == 0:
            is_tr = X["year"] < val_period
        else:
            prev_val_period = _val_period_list[i - 1]
            is_tr = (X["year"] < val_period) & (X["year"] >= prev_val_period)

        is_val = X["year"] == val_period
        tr_x, val_x = X[is_tr], X[is_val]
        tr_y, val_y = y[is_tr], y[is_val]

        dtrain, dval = (
            lgb.Dataset(
                data=tr_x,
                label=tr_y,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
            lgb.Dataset(
                data=val_x,
                label=val_y,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
        )

        # 訓練
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
        )
        # 検証
        y_pred_train = gbm.predict(tr_x)
        y_pred_val = gbm.predict(val_x)

        tr_rmse = np.sqrt(mean_squared_error(tr_y, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(val_y, y_pred_val))
        # 検証の格納
        rmse_scores.append(val_rmse)

        print(
            f"Fold: {len(rmse_scores) + 1}   Train RMSLE: {tr_rmse:.4f}  Val RMSLE: {val_rmse:.4f}"
        )
        print("-" * 50)

    return rmse_scores, gbm


def time_cross_val(X: pd.DataFrame, y: pd.DataFrame, params: dict, config: CFG):
    rmse_scores = []

    # 年ごとにデータの分割
    for val_period in _val_period_list:
        is_tr = X["year"] < val_period
        is_val = X["year"] == val_period
        tr_x, val_x = X[is_tr], X[is_val]
        tr_y, val_y = y[is_tr], y[is_val]

        dtrain, dval = (
            lgb.Dataset(
                data=tr_x,
                label=tr_y,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
            lgb.Dataset(
                data=val_x,
                label=val_y,
                feature_name=config.FEATURES,
                categorical_feature=config.CAT_FEATURES,
            ),
        )

        # 訓練
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=15)],
        )
        # 検証
        y_pred_train = gbm.predict(tr_x)
        y_pred_val = gbm.predict(val_x)

        tr_rmse = np.sqrt(mean_squared_error(tr_y, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(val_y, y_pred_val))
        # 検証の格納
        rmse_scores.append(val_rmse)

        print(
            f"Fold: {len(rmse_scores) + 1}   Train RMSLE: {tr_rmse:.4f}  Val RMSLE: {val_rmse:.4f}"
        )
        print("-" * 50)

    return rmse_scores
