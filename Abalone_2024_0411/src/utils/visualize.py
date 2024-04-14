import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union, Any, Optional


def plot_heatmap(
    df: pd.DataFrame,
    numeric_vars: list[str],
    cmap: Optional[str] = "coolwarm",
    center: Optional[int | float] = 0.75,
    annnot: Optional[bool] = True,
) -> None:
    # 相関係数の計算
    cc = np.corrcoef(df[numeric_vars], rowvar=False)
    sns.heatmap(
        cc,
        center=center,
        cmap=cmap,
        annot=annnot,
        xticklabels=numeric_vars,
        yticklabels=numeric_vars,
    )
    plt.show()


def plot_numerics(
    df: pd.DataFrame,
    numeric_vars: list[str],
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


def plot_pie(df: pd.DataFrame, categorical_vars: list[str]) -> None:
    for col in categorical_vars:
        counts = df[col].value_counts()

        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.pie(counts, labels=counts.index, autopct="%.1f%%", startangle=90)
        axes.set_title(f"Pie gragh of {col}")
        plt.show()


def plot_counts(df: pd.DataFrame, categorical_vars: list[str]) -> None:
    for col in categorical_vars:
        fig, axes = plt.subplots(1, 1, figsize=(7, 4))

        sns.countplot(
            x=df[col], order=df[col].value_counts(ascending=False).index, ax=axes
        )

        abs_values = df[col].value_counts(ascending=False)
        rel_values = df[col].value_counts(ascending=False, normalize=True).values * 100
        lbls = [f"{p[0]}  ({p[1]:.1f}%)" for p in zip(abs_values, rel_values)]
        axes.bar_label(axes.containers[0], labels=lbls)
        axes.set_title(f"Distribution of {col}")
        plt.show()
