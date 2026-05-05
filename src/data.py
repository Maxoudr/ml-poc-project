from __future__ import annotations
from typing import Any
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    
    return X_train, X_test, y_train, y_test