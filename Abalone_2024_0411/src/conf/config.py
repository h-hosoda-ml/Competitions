import os
import sys

from dotenv import load_dotenv

load_dotenv()

DIRS = {
    "BASE": os.getenv("BASE_DIR", ".envnone"),
    "DATA": os.getenv("DATA_DIR", ".envnone"),
    "SRC": os.getenv("SRC_DIR", ".envnone"),
    "OUTPUT": os.getenv("OUTPUT_DIR", ".env none"),
}

PARAMS = {"seed": 42}


class CFG:
    ID = "id"
    TARGET = "Rings"
    CATEGORIES = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Whole weight.1",
        "Whole weight.2",
        "Shell weight",
    ]

    NUMERIC_VARS = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Whole weight.1",
        "Whole weight.2",
        "Shell weight",
        "Rings",
    ]

    NUMERIC_FEATURES = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Whole weight.1",
        "Whole weight.2",
        "Shell weight",
    ]

    CAT_FEATURES = ["Sex"]

    # testの設定
    TEST_SIZE = 0.25
    SHUFFLE = True
