import os
import sys
import re

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf.config import CFG, DIRS

_alphabet_table = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
)

_city_pattern = re.compile(r"(?P<prefecture>.+?[都道府県])(?P<city>.+?[市区町村郡])")


def _extract_broadcast(row):
    is_recorded = 0
    broadcast_type = "地上波"

    broadcasts = row.split("/")

    for b in broadcasts:
        # 録画放送
        if "(録)" in b:
            is_recorded = 1

        # 放送タイプ
        # 放送タイプのチェック
        if "BS" in b:
            broadcast_type = "BS"
        elif "スカパー!" in b or "J SPORTS" in b:
            broadcast_type = "CS"
        elif "MXテレビ" in b:
            broadcast_type = "インターネット配信"

    return pd.Series([is_recorded, broadcast_type])


def _extract_components(address):
    match = _city_pattern.match(address)
    if match:
        return match.groupdict()
    return {"prefecture": None, "city": None}


def _aggredate_weather(value):
    if "屋内" in value:
        return "屋内"
    elif value == "雷" or value == "霧":
        return "雨"
    elif len(value) == 1:
        return value
    else:
        return value[0]


def feature_engineering_prototype2(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: CFG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # home_team
    train_df.loc[:, "home_team"] = train_df.loc[:, "home_team"].apply(
        lambda text: text.translate(_alphabet_table)
    )
    test_df.loc[:, "home_team"] = test_df.loc[:, "home_team"].apply(
        lambda text: text.translate(_alphabet_table)
    )

    # away_team
    train_df.loc[:, "away_team"] = train_df.loc[:, "away_team"].apply(
        lambda text: text.translate(_alphabet_table)
    )
    test_df.loc[:, "away_team"] = test_df.loc[:, "away_team"].apply(
        lambda text: text.translate(_alphabet_table)
    )

    # match_date
    train_df["match_date"] = pd.to_datetime(train_df["match_date"])
    test_df["match_date"] = pd.to_datetime(test_df["match_date"])

    # 年
    train_df["year"] = train_df["match_date"].dt.year
    test_df["year"] = test_df["match_date"].dt.year
    # 月
    train_df["month"] = train_df["match_date"].dt.month
    test_df["month"] = test_df["match_date"].dt.month
    # 曜日
    train_df["weekday"] = train_df["match_date"].dt.day_name()
    test_df["weekday"] = test_df["match_date"].dt.day_name()
    # 祝日フラグ
    # データのロード
    holiday_df = pd.read_csv(os.path.join(DIRS["DATA"], "holidays_in_japan.csv"))
    holiday_df["holiday_date"] = pd.to_datetime(holiday_df["holiday_date"])

    # 結合
    train_df = pd.merge(
        train_df, holiday_df, left_on="match_date", right_on="holiday_date", how="left"
    )
    test_df = pd.merge(
        test_df, holiday_df, left_on="match_date", right_on="holiday_date", how="left"
    )
    # 0, 1 フラグに変換
    train_df["is_holiday"] = train_df["description"].apply(
        lambda val: 0 if pd.isna(val) else 1
    )
    test_df["is_holiday"] = test_df["description"].apply(
        lambda val: 0 if pd.isna(val) else 1
    )

    # kick_off_time
    # datetimeオブジェクトに変換
    train_df["kick_off_time"] = pd.to_datetime(
        train_df["kick_off_time"], format="%H:%M"
    )
    test_df["kick_off_time"] = pd.to_datetime(test_df["kick_off_time"], format="%H:%M")

    # 時間
    train_df["hour"] = train_df["kick_off_time"].dt.hour
    test_df["hour"] = test_df["kick_off_time"].dt.hour

    # broadcasters
    train_df[["is_recorded", "broadcast_type"]] = train_df["broadcasters"].apply(
        _extract_broadcast
    )
    test_df[["is_recorded", "broadcast_type"]] = train_df["broadcasters"].apply(
        _extract_broadcast
    )

    # address
    component_train_df = (
        train_df.loc[:, "address"].apply(_extract_components).apply(pd.Series)
    )
    component_test_df = (
        test_df.loc[:, "address"].apply(_extract_components).apply(pd.Series)
    )
    train_df = pd.concat([train_df, component_train_df], axis=1)
    test_df = pd.concat([test_df, component_test_df], axis=1)

    # 必要なカラムのみ抽出し、return
    feature_col_names = [
        "section",
        "round",
        "home_team",
        "away_team",
        "temperature",
        "humidity",
        "home_team_score",
        "away_team_score",
        "capacity",
        "year",
        "month",
        "weekday",
        "is_holiday",
        "hour",
        "is_recorded",
        "broadcast_type",
        "prefecture",
        "city",
    ]
    result_train = train_df.loc[:, ["id"] + feature_col_names + ["attendance"]]
    result_test = test_df.loc[:, ["id"] + feature_col_names]
    # configの書き換え
    config.FEATURES = feature_col_names
    config.CAT_FEATURES = [
        "section",
        "round",
        "home_team",
        "away_team",
        "weekday",
        "hour",
        "prefecture",
        "city",
        "broadcast_type",
        "is_holiday",
        "is_recorded",
    ]
    config.NUM_FEATURES = [
        "temperature",
        "humidity",
        "home_team_score",
        "away_team_score",
        "capacity",
        "year",
        "month",
    ]
    # object -> categoryへ変換
    result_train[config.CAT_FEATURES] = result_train[config.CAT_FEATURES].astype(
        "category"
    )
    result_test[config.CAT_FEATURES] = result_test[config.CAT_FEATURES].astype(
        "category"
    )

    return (result_train, result_test)
