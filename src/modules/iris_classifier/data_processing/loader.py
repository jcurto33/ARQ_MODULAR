"""
src.modules.iris_classifier.data_processing.loader
────────────────────────────────────────────────────
Carga y preparación de datos Iris.
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from shared.config_loader import load_config

_config = load_config()
_RAW_DIR = Path(_config["paths"]["raw_dir"])
_SPLITS_DIR = Path(_config["paths"]["data_dir"]) / "files" / "splits"


def download_iris() -> Path:
    """Descarga el dataset Iris y lo guarda como CSV en data/files/raw/."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _RAW_DIR / "iris.csv"

    iris = load_iris(as_frame=True)
    df = iris.frame  # type: ignore
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))  # type: ignore
    df.to_csv(path, index=False)
    return path


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga el CSV y devuelve (X_train, X_test, y_train, y_test)."""
    csv_path = _RAW_DIR / "iris.csv"
    if not csv_path.exists():
        download_iris()

    df = pd.read_csv(csv_path)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["target"]

    test_size = _config["iris_classifier"]["test_size"]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
