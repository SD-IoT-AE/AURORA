
"""AURORA: Weighted RF/XGBoost/LightGBM/GRU ensemble."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)

from aurora.models.classification import ThreatClassifier
from aurora.models.gru import GRUClassifier


class AURORAEnsemble:
    """Four-model behavioral threat detection ensemble."""

    WEIGHTS = {
        "rf": 0.20,
        "xgb": 0.25,
        "lgbm": 0.25,
        "gru": 0.30,
    }

    def __init__(self, input_dim, seed=42, gru_epochs=20):
        self.models = {
            name: ThreatClassifier(name, seed)
            for name in ("rf", "xgb", "lgbm")
        }
        self.models["gru"] = GRUClassifier(
            input_dim=input_dim,
            epochs=gru_epochs,
            batch_size=32,
            learning_rate=0.001,
            dropout=0.2,
            seed=seed,
        )

    @staticmethod
    def _tabular(X):
        """Use the last time step for tree-based models."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            return X[:, -1, :]
        if X.ndim == 2:
            return X
        raise ValueError("Expected 2D or 3D features")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all four models on training data only."""
        X_tab = self._tabular(X_train)

        for name in ("rf", "xgb", "lgbm"):
            self.models[name].fit(X_tab, y_train)

        self.models["gru"].fit(
            X_train, y_train, X_val, y_val
        )
        return self

    def predict_components(self, X):
        """Return individual malicious-class probabilities."""
        X_tab = self._tabular(X)
        return {
            name: np.asarray(
                model.predict_proba(
                    X if name == "gru" else X_tab
                ),
                dtype=float,
            ).reshape(-1)
            for name, model in self.models.items()
        }

    def predict_proba(self, X):
        """Weighted probability of malicious behavior."""
        components = self.predict_components(X)
        return np.clip(
            sum(
                self.WEIGHTS[name] * components[name]
                for name in self.WEIGHTS
            ),
            0.0, 1.0,
        )

    def predict(self, X, threshold=0.5):
        return (
            self.predict_proba(X) >= threshold
        ).astype(int)

    def evaluate(self, X_test, y_test):
        """Calculate ensemble classification metrics."""
        y = np.asarray(y_test, dtype=int)
        p = self.predict_proba(X_test)

        if y.shape != p.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid evaluation labels")

        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        return {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(
                y, pred, zero_division=0
            )),
            "recall": float(recall_score(
                y, pred, zero_division=0
            )),
            "f1_score": float(f1_score(
                y, pred, zero_division=0
            )),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "roc_auc": (
                float(roc_auc_score(y, p))
                if len(np.unique(y)) == 2 else None
            ),
        }

    def assess_uav(self, trust_manager, uav_id, X, timestamp):
        """Update TOSA trust from one UAV observation."""
        probabilities = self.predict_proba(X)

        if len(probabilities) != 1:
            raise ValueError("Provide exactly one UAV observation")

        return trust_manager.update_from_probability(
            uav_id=uav_id,
            malicious_probability=float(probabilities[0]),
            timestamp=timestamp,
        )


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from aurora.tosa.trust import TrustManager

    rng = np.random.default_rng(42)

    # Synthetic sequences: samples x time steps x features
    X = rng.normal(size=(1000, 8, 12)).astype(np.float32)
    y = (
        X[:, :, 0].mean(axis=1)
        + X[:, -1, 1] > 0
    ).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    ensemble = AURORAEnsemble(
        input_dim=12, seed=42, gru_epochs=5
    )

    ensemble.fit(X_train, y_train)

    print("Ensemble metrics:", ensemble.evaluate(X_test, y_test))

    trust_manager = TrustManager()
    observation = ensemble.assess_uav(
        trust_manager,
        uav_id=1,
        X=X_test[:1],
        timestamp=1.0,
    )

    print(
        f"UAV 1 | Trust={observation.trust:.4f} | "
        f"Risk={observation.risk:.4f} | "
        f"FL eligible={observation.eligible_for_fl}"
    )
