
"""AURORA: Two-layer GRU behavioral threat classifier."""

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)


class GRUNetwork(nn.Module):
    """Two stacked GRU layers: 64 and 32 hidden units."""

    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 32, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dropout(x[:, -1, :])
        return self.output(x).squeeze(-1)


class GRUClassifier:
    """Training and inference interface for AURORA GRU."""

    def __init__(
        self, input_dim, epochs=20, batch_size=32,
        learning_rate=0.001, dropout=0.2,
        seed=42, device="cpu",
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = GRUNetwork(input_dim, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = []

    @staticmethod
    def _prepare(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]  # Single-step sequence
        if X.ndim != 3 or not np.isfinite(X).all():
            raise ValueError("Expected finite [N, time, features]")
        if X.shape[0] == 0 or min(X.shape[1:]) < 1:
            raise ValueError("Sequence dimensions must be positive")
        return torch.from_numpy(X)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train using labeled sequences; optionally evaluate validation loss."""
        X = self._prepare(X_train)
        y = np.asarray(y_train, dtype=np.float32)

        if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be binary and match X")

        loader = DataLoader(
            TensorDataset(X, torch.from_numpy(y)),
            batch_size=self.batch_size, shuffle=True,
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(xb)

            record = {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(X),
            }

            if X_val is not None and y_val is not None:
                xv = self._prepare(X_val).to(self.device)
                yv = torch.as_tensor(
                    y_val, dtype=torch.float32, device=self.device
                )
                if yv.shape != (len(xv),):
                    raise ValueError("Invalid validation labels")
                self.model.eval()
                with torch.no_grad():
                    record["val_loss"] = self.criterion(
                        self.model(xv), yv
                    ).item()

            self.history.append(record)

        return self

    def predict_proba(self, X):
        """Return P(malicious) for each sequence."""
        X = self._prepare(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(X))
        return probabilities.cpu().numpy()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """Evaluate classification performance on held-out data."""
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

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        return self


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic temporal sequences: [samples, time, features]
    X = rng.normal(size=(1000, 10, 12)).astype(np.float32)
    y = (X[:, :, 0].mean(axis=1) +
         X[:, :, 1].mean(axis=1) > 0).astype(int)

    model = GRUClassifier(
        input_dim=12, epochs=5, batch_size=32,
        learning_rate=0.001, dropout=0.2,
    )

    model.fit(X[:700], y[:700], X[700:850], y[700:850])
    print(model.evaluate(X[850:], y[850:]))
