
"""AURORA: RF, XGBoost, and LightGBM threat classifiers."""

from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ThreatClassifier:
    """Unified interface for AURORA behavioral classifiers."""

    def __init__(self, model_type="rf", seed=42):
        self.model_type = model_type.lower()
        self.seed = seed

        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        elif self.model_type == "xgb":
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
            )
        elif self.model_type == "lgbm":
            self.model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
        else:
            raise ValueError("Choose rf, xgb, or lgbm")

    @staticmethod
    def _validate(X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array")

        if y is None:
            return X

        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Invalid labels")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be binary: 0 or 1")

        return X, y.astype(int)

    def fit(self, X_train, y_train):
        """Train using training-set data only."""
        X, y = self._validate(X_train, y_train)
        if len(np.unique(y)) != 2:
            raise ValueError("Both classes required for training")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(self._validate(X))

    def predict_proba(self, X):
        """Return P(malicious), where malicious label = 1."""
        X = self._validate(X)
        probabilities = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        return probabilities[:, classes.index(1)]

    def evaluate(self, X_test, y_test):
        """Evaluate using independent held-out data."""
        X, y = self._validate(X_test, y_test)
        pred = self.predict(X)
        prob = self.predict_proba(X)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0
            )
        )
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        return {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "roc_auc": (
                float(roc_auc_score(y, prob))
                if len(np.unique(y)) == 2 else None
            ),
            "confusion_matrix": [[int(tn), int(fp)],
                                 [int(fn), int(tp)]],
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "seed": self.seed,
            "model": self.model,
        }, path)

    @classmethod
    def load(cls, path):
        # Load only files from trusted sources.
        data = joblib.load(path)
        instance = cls(data["model_type"], data["seed"])
        instance.model = data["model"]
        return instance


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    for name in ("rf", "xgb", "lgbm"):
        classifier = ThreatClassifier(name, seed=42)
        classifier.fit(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)

        print(
            f"{name.upper():5s} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"FPR={metrics['fpr']:.4f}"
        )
