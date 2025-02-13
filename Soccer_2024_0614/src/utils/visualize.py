import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

from typing import Union, Optional

"""数値変数に対する関数群"""


def plot_num_target_distribution(
    df: pd.DataFrame, numerical_var: Union[list[str], str], target_var: str
):
    if isinstance(numerical_var, str):
        numerical_var = [numerical_var]

    for col in numerical_var:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        sns.boxenplot(data=df, x=col, y=target_var)
        ax.set_title(f"Distribuyion of {col}")

        plt.show()


def plot_num_distribution(
    df: pd.DataFrame,
    numeric_vars: Union[list[str], str],
) -> None:
    for col in numeric_vars:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), tight_layout=True)
        axes = axes.flatten()

        # Histgram
        sns.histplot(df, x=col, kde=False, ax=axes[0], bins=40)
        axes[0].set_title(f"Hisgram of {col}", fontweight="bold")

        # Box Plot
        sns.boxplot(df, x=col, color="lightgreen", ax=axes[1])
        axes[1].set_title(f"Box Plot of {col}", fontweight="bold")

        # 描画
        plt.show()


"""カテゴリ変数に対する関数群"""


def plot_cat_counts(
    df: pd.DataFrame,
    categorical_var: Union[list[str], str],
    need_percent: Optional[bool] = False,
):
    """カテゴリ変数のカテゴリごとのカウント数をプロットする関数

    Args:
        df (pd.DataFrame): 描画を行いたいDataFrameオブジェクト
        categorical_var (Union[list[str], str]): 描画を行いたいカテゴリ変数名
        need_percent (Optional[bool], optional): %表示が必要か Defaults to False.
    """
    if isinstance(categorical_var, str):
        categorical_var = [categorical_var]

    for col in categorical_var:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        sns.countplot(
            x=df[col],
            order=df[col].value_counts(ascending=False).index,
            ax=ax,
        )
        if need_percent:
            abs_values = df[col].value_counts(ascending=False)
            rel_values = (
                df[col].value_counts(ascending=False, normalize=True).values * 100
            )
            lbls = [f"{p[0]}  ({p[1]:.1f}%)" for p in zip(abs_values, rel_values)]
            ax.bar_label(ax.containers[0], labels=lbls)
        ax.set_title(f"Distribuyion of {col}")
        plt.xticks(rotation=90)
        plt.show()


def plot_cat_distribution(
    df: pd.DataFrame, categorical_var: Union[list[str], str], target_var: str
):
    if isinstance(categorical_var, str):
        categorical_var = [categorical_var]

    for col in categorical_var:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        sns.boxenplot(x=col, y=target_var, data=df)
        ax.set_title(f"Distribuyion of {col}")
        plt.xticks(rotation=90)
