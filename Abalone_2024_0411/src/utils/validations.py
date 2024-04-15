import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import *


def cross_val_train(X: pd.DataFrame, y: pd.DataFrame, df_test, params):
    kf = KFold(n_splits=5, shuffle=True, random_state=PARAMS["seed"])
    test_preds = np.zeros((len(df_test)))
    val_preds = np.zeros((len(X)))
    val_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_train = X.iloc[tr_idx]
        y_train = y[tr_idx]
        X_val = X.iloc[va_idx]
        y_val = y[va_idx]

        dtrain, dval = (
            lgb.Dataset(data=X_train, label=y_train),
            lgb.Dataset(data=X_val, label=y_val),
        )

        LGB = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=5000,
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

    return val_scores, val_preds, test_preds
