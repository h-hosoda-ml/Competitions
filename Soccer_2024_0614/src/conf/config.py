import os
import sys

from dotenv import load_dotenv

load_dotenv()

DIRS = {
    "BASE": os.getenv("BASE_DIR", ".env none"),
    "DATA": os.getenv("DATA_DIR", ".envnone"),
    "SRC": os.getenv("SRC_DIR", ".envnone"),
    "OUTPUT": os.getenv("OUTPUT_DIR", ".env none"),
}


class CFG:
    # データについて
    ID: str = "id"
    CATEGORIES: list = []
    NUMERIC_VARS: list = []

    # 特徴量について
    FEATURES: list = []
    TARGET: str = "attendance"
    CAT_FEATURES: list = []
    NUM_FEATURES: list = []

    # seedについて
    SEED = 42

    # testの設定
    TEST_SIZE = 0.25
    PERIOD = []
    SHUFFLE = True
