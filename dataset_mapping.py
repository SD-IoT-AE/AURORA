
"""AURORA: Dataset harmonization and leakage-safe preprocessing."""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetConfig:
    label_column: str = "label"
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
    max_features: int = 50
    group_column: str = ""
    benign_labels: tuple = (
        "benign", "normal", "normal traffic",
        "0", "non-attack",
    )


class DatasetMapper:
    """Prepare consistent binary intrusion-detection datasets."""

    DATASETS = {
        "ciciot2023": "CICIoT2023",
        "edgeiiot": "Edge-IIoTset",
        "toniot": "ToN_IoT",
        "custom": "Custom",
    }

    def __init__(self, config=None):
        self.config = config or DatasetConfig()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.features = []

    def load(self, path, dataset="custom"):
        """Load CSV or Parquet data; preserve original labels."""
        if dataset.lower() not in self.DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        path = Path(path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, low_memory=False)
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError("Supported formats: CSV, Parquet")

        if self.config.label_column not in frame:
            raise KeyError("Label column not found")

        return frame

    def map_labels(self, labels):
        """Map benign=0, malicious=1; reject missing labels."""
        values = pd.Series(labels)

        if values.isna().any():
            raise ValueError("Missing ground-truth labels")

        benign = {
            str(x).strip().lower()
            for x in self.config.benign_labels
        }

        mapped = values.astype(str).str.strip().str.lower()
        return (~mapped.isin(benign)).astype(np.int64).to_numpy()

    def split(self, frame):
        """Split before fitting preprocessing transformations."""
        cfg = self.config
        y = self.map_labels(frame[cfg.label_column])

        if cfg.group_column:
            if cfg.group_column not in frame:
                raise KeyError("Group column not found")

            groups = frame[cfg.group_column].to_numpy()
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=cfg.test_size,
                random_state=cfg.seed,
            )
            train_val_idx, test_idx = next(
                splitter.split(frame, y, groups)
            )

            relative_val = cfg.val_size / (1 - cfg.test_size)
            second = GroupShuffleSplit(
                n_splits=1, test_size=relative_val,
                random_state=cfg.seed + 1,
            )
            train_rel, val_rel = next(second.split(
                frame.iloc[train_val_idx],
                y[train_val_idx],
                groups[train_val_idx],
            ))

            train_idx = train_val_idx[train_rel]
            val_idx = train_val_idx[val_rel]
        else:
            indices = np.arange(len(frame))
            train_val_idx, test_idx = train_test_split(
                indices, test_size=cfg.test_size,
                random_state=cfg.seed, stratify=y,
            )
            relative_val = cfg.val_size / (1 - cfg.test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=relative_val,
                random_state=cfg.seed + 1,
                stratify=y[train_val_idx],
            )

        return (
            frame.iloc[train_idx].copy(),
            frame.iloc[val_idx].copy(),
            frame.iloc[test_idx].copy(),
        )

    def fit_transform(self, train, val, test):
        """Fit imputation/scaling on training data only."""
        cfg = self.config
        excluded = {cfg.label_column, cfg.group_column}

        numeric = train.select_dtypes(
            include=[np.number]
        ).columns

        self.features = [
            col for col in numeric
            if col not in excluded
        ][:cfg.max_features]

        if not self.features:
            raise ValueError("No numeric features available")

        def extract(frame):
            X = frame.reindex(columns=self.features)
            X = X.replace([np.inf, -np.inf], np.nan)
            return X.to_numpy(dtype=np.float64)

        X_train = extract(train)
        X_val = extract(val)
        X_test = extract(test)

        X_train = self.imputer.fit_transform(X_train)
        X_val = self.imputer.transform(X_val)
        X_test = self.imputer.transform(X_test)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return {
            "X_train": X_train.astype(np.float32),
            "y_train": self.map_labels(train[cfg.label_column]),
            "X_val": X_val.astype(np.float32),
            "y_val": self.map_labels(val[cfg.label_column]),
            "X_test": X_test.astype(np.float32),
            "y_test": self.map_labels(test[cfg.label_column]),
            "feature_names": self.features,
        }

    def prepare(self, path, dataset="custom"):
        """Execute the complete dataset preparation pipeline."""
        frame = self.load(path, dataset)
        train, val, test = self.split(frame)
        return self.fit_transform(train, val, test)

    def save(self, data, output_dir):
        """Export arrays and feature metadata."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path / "dataset.npz",
            **{k: v for k, v in data.items()
               if isinstance(v, np.ndarray)},
        )

        with (path / "features.json").open("w") as f:
            json.dump(data["feature_names"], f, indent=2)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic demonstration; not benchmark data.
    frame = pd.DataFrame(
        rng.normal(size=(2000, 12)),
        columns=[f"feature_{i}" for i in range(12)],
    )
    frame["label"] = np.where(
        frame["feature_0"] + frame["feature_1"] > 0,
        "Attack", "Benign",
    )

    mapper = DatasetMapper()
    train, val, test = mapper.split(frame)
    data = mapper.fit_transform(train, val, test)
    mapper.save(data, "outputs/datasets/demo")

    print("Training:", data["X_train"].shape)
    print("Validation:", data["X_val"].shape)
    print("Testing:", data["X_test"].shape)
