import os
import sys

from dotenv import load_dotenv

load_dotenv()

DIRS = {
    "BASE": os.getenv("BASE_DIR", ".envnone"),
    "DATA": os.getenv("DATA_DIR", ".envnone"),
    "SRC": os.getenv("SRC_DIR", ".envnone"),
}

PARAMS = {"seed": 42}
