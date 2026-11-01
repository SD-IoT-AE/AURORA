
"""
AURORA - Federated Learning Training Workflow
==============================================

Module:
    Risk-Aware Federated Learning Decision (RFLD)

Purpose:
    Execute reproducible federated-learning experiments
    using trust-aware client selection and aggregation.

Features:
    - 50 UAV clients by default
    - 20 selected clients per FL round
    - Dirichlet non-IID data partitioning
    - Local PyTorch model training
    - Adam optimizer and cross-entropy loss
    - Configurable model-update poisoning
    - TOSA trust-based participation control
    - AURORA, FedAvg, Krum, Trimmed Mean, Median
    - Held-out validation and test evaluation
    - Per-round JSON/CSV logging
    - Model checkpoint saving
    - Reproducible random seeds

Default configuration:
    rounds = 50
    clients_per_round = 20
    local_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    malicious_fraction = 0.20
    poisoning_strength = 0.20
    trust_threshold = 0.50
    dirichlet_alpha = 0.50
    seed = 42

IMPORTANT:
    This module implements a reproducible reference
    FL workflow.

    It does not reproduce manuscript numerical
    results without the original datasets,
    preprocessing, trust observations, and
    experimental configurations.

    Synthetic examples are used only for
    functional demonstration.
"""

from __future__ import annotations

import csv
import json
import random

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from aurora.tosa.trust import (
    TrustConfig,
    TrustManager,
)

from aurora.rfld.partition import (
    FederatedDataPartitioner,
    PartitionConfig,
    PartitionResult,
)

from aurora.rfld.aggregation import (
    AggregationConfig,
    ClientUpdate,
    FederatedAggregator,
)


@dataclass(frozen=True)
class FLConfig:
    """Federated-learning experiment configuration."""

    num_clients: int = 50
    clients_per_round: int = 20

    rounds: int = 50
    local_epochs: int = 5

    batch_size: int = 32
    learning_rate: float = 0.001

    hidden_dim: int = 64

    dirichlet_alpha: float = 0.50
    min_samples_per_client: int = 10

    malicious_fraction: float = 0.20
    poisoning_strength: float = 0.20

    trust_threshold: float = 0.50
    trust_alpha: float = 0.70

    aggregation_method: str = "aurora"

    seed: int = 42
    device: str = "cpu"

    output_dir: str = "outputs/fl"

    def __post_init__(self) -> None:

        positive_integers = (
            "num_clients",
            "clients_per_round",
            "rounds",
            "local_epochs",
            "batch_size",
            "hidden_dim",
        )

        for name in positive_integers:
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be positive."
                )

        if self.clients_per_round > self.num_clients:
            raise ValueError(
                "clients_per_round cannot exceed "
                "num_clients."
            )

        if not np.isfinite(self.learning_rate):
            raise ValueError(
                "learning_rate must be finite."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.dirichlet_alpha <= 0:
            raise ValueError(
                "dirichlet_alpha must be positive."
            )

        if self.min_samples_per_client < 1:
            raise ValueError(
                "min_samples_per_client must be >= 1."
            )

        for name in (
            "malicious_fraction",
            "poisoning_strength",
            "trust_threshold",
            "trust_alpha",
        ):

            value = getattr(self, name)

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be within [0, 1]."
                )

        supported = {
            "aurora",
            "fedavg",
            "krum",
            "trimmed_mean",
            "median",
        }

        if self.aggregation_method not in supported:
            raise ValueError(
                "Unsupported aggregation method."
            )


class TabularFLModel(nn.Module):
    """
    Lightweight neural network for tabular
    federated-learning experiments.

    Architecture:
        Input
        Linear(hidden_dim)
        ReLU
        Linear(hidden_dim // 2)
        ReLU
        Linear(num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                max(1, hidden_dim // 2),
            ),
            nn.ReLU(),
            nn.Linear(
                max(1, hidden_dim // 2),
                num_classes,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(features)


def set_global_seed(
    seed: int,
) -> None:
    """Configure reproducible random seeds."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_dataset(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate a tabular dataset.

    Features must already be:
        - Numeric
        - Encoded
        - Imputed
        - Normalized using training-only statistics

    Labels must be integer class indices.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError(
            "Features must be a 2D array."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Labels must be a 1D array."
        )

    if len(features) != len(labels):
        raise ValueError(
            "Feature and label counts differ."
        )

    if len(features) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Features must be finite."
        )

    if not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "Labels must be integer encoded."
        )

    labels = labels.astype(
        np.int64,
        copy=False,
    )

    if np.any(labels < 0):
        raise ValueError(
            "Labels must be nonnegative."
        )

    return features, labels


def get_model_parameters(
    model: nn.Module,
) -> List[np.ndarray]:
    """
    Extract trainable parameters in model order.

    This reference model contains only trainable
    parameters, so no buffer synchronization is
    required.
    """

    return [
        parameter.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .copy()
        for parameter in model.parameters()
    ]


def set_model_parameters(
    model: nn.Module,
    parameters: Sequence[np.ndarray],
) -> None:
    """Assign aggregated parameters to a model."""

    model_parameters = list(
        model.parameters()
    )

    if len(model_parameters) != len(parameters):
        raise ValueError(
            "Model parameter count mismatch."
        )

    with torch.no_grad():

        for target, source in zip(
            model_parameters,
            parameters,
        ):

            source_array = np.asarray(
                source
            )

            if tuple(target.shape) != tuple(
                source_array.shape
            ):
                raise ValueError(
                    "Model parameter shape mismatch."
                )

            if not np.all(
                np.isfinite(source_array)
            ):
                raise ValueError(
                    "Aggregated parameters are not finite."
                )

            tensor = torch.as_tensor(
                source_array,
                dtype=target.dtype,
                device=target.device,
            )

            target.copy_(tensor)


def evaluate_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Evaluate model accuracy and cross-entropy loss.

    This function does not perform training.
    """

    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += int(
                (predictions == batch_y)
                .sum()
                .item()
            )

            total += len(batch_y)

            total_loss += (
                float(loss.item())
                * len(batch_y)
            )

    if total == 0:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    return {
        "accuracy": 100.0 * correct / total,
        "loss": total_loss / total,
        "num_samples": total,
    }


def train_local_model(
    global_parameters: Sequence[np.ndarray],
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
) -> Tuple[List[np.ndarray], float]:
    """
    Train one UAV client's local model.

    Returns:
        Updated model parameters
        Mean local training loss
    """

    torch.manual_seed(seed)

    model = TabularFLModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    ).to(device)

    set_model_parameters(
        model,
        global_parameters,
    )

    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            total_batches += 1

    mean_loss = (
        total_loss / total_batches
        if total_batches
        else 0.0
    )

    return (
        get_model_parameters(model),
        mean_loss,
    )


def apply_model_update_poisoning(
    global_parameters: Sequence[np.ndarray],
    local_parameters: Sequence[np.ndarray],
    strength: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Simulate additive model-update poisoning.

    Define the original client update:
        delta = theta_local - theta_global

    Generate a random perturbation with norm:
        ||epsilon|| = strength * ||delta||

    Submit:
        theta_poisoned = theta_local + epsilon

    The perturbation direction is sampled from
    a standard normal distribution.

    This models norm-bounded random update
    corruption, not an optimized Byzantine attack.
    """

    if not 0 <= strength <= 1:
        raise ValueError(
            "Poisoning strength must be within [0, 1]."
        )

    if len(global_parameters) != len(
        local_parameters
    ):
        raise ValueError(
            "Parameter count mismatch."
        )

    original_updates = [
        np.asarray(local, dtype=np.float64)
        - np.asarray(global_, dtype=np.float64)
        for global_, local in zip(
            global_parameters,
            local_parameters,
        )
    ]

    update_vector = np.concatenate(
        [
            update.ravel()
            for update in original_updates
        ]
    )

    update_norm = float(
        np.linalg.norm(update_vector)
    )

    if strength == 0 or update_norm == 0:
        return [
            np.asarray(parameter).copy()
            for parameter in local_parameters
        ]

    direction = rng.normal(
        size=update_vector.shape
    )

    direction_norm = float(
        np.linalg.norm(direction)
    )

    if direction_norm == 0:
        return [
            np.asarray(parameter).copy()
            for parameter in local_parameters
        ]

    perturbation = (
        direction / direction_norm
        * strength
        * update_norm
    )

    poisoned_parameters = []

    offset = 0

    for parameter in local_parameters:

        array = np.asarray(
            parameter,
            dtype=np.float64,
        )

        size = array.size

        noise = perturbation[
            offset:offset + size
        ].reshape(array.shape)

        poisoned_parameters.append(
            array + noise
        )

        offset += size

    return poisoned_parameters


class AURORAFederatedTrainer:
    """
    Main AURORA federated-learning coordinator.

    Integrates:
        - Dirichlet client partitions
        - Trust-based client selection
        - Local neural-network training
        - Simulated model-update poisoning
        - Federated aggregation
        - Evaluation and logging

    Trust updates are supplied externally by TOSA
    or through an explicit trust-observation mapping.

    The coordinator does not infer maliciousness
    from ground-truth attacker labels.
    """

    def __init__(
        self,
        config: Optional[FLConfig] = None,
        trust_manager: Optional[TrustManager] = None,
    ) -> None:

        self.config = config or FLConfig()

        set_global_seed(
            self.config.seed
        )

        self.device = torch.device(
            self.config.device
        )

        self.trust_manager = (
            trust_manager
            if trust_manager is not None
            else TrustManager(
                TrustConfig(
                    alpha=self.config.trust_alpha,
                    trust_threshold=(
                        self.config.trust_threshold
                    ),
                )
            )
        )

        if not np.isclose(
            self.trust_manager.config.trust_threshold,
            self.config.trust_threshold,
        ):
            raise ValueError(
                "Trust manager and FL configuration "
                "must use the same trust threshold."
            )

        self.partitioner = FederatedDataPartitioner(
            PartitionConfig(
                num_clients=self.config.num_clients,
                dirichlet_alpha=(
                    self.config.dirichlet_alpha
                ),
                min_samples_per_client=(
                    self.config.min_samples_per_client
                ),
                clients_per_round=(
                    self.config.clients_per_round
                ),
                seed=self.config.seed,
            )
        )

        self.aggregator = FederatedAggregator(
            AggregationConfig(
                trust_threshold=(
                    self.config.trust_threshold
                ),
                krum_byzantine_clients=2,
                trimmed_mean_fraction=0.20,
            )
        )

        self.output_dir = Path(
            self.config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history: List[Dict] = []

        self.malicious_clients: set[int] = set()

        self.partition: Optional[
            PartitionResult
        ] = None

        self.model: Optional[
            TabularFLModel
        ] = None

    def initialize_clients(
        self,
    ) -> None:
        """Register all UAVs with the trust manager."""

        for client_id in range(
            self.config.num_clients
        ):

            self.trust_manager.register_uav(
                client_id
            )

    def assign_malicious_clients(
        self,
    ) -> set[int]:
        """
        Select a fixed malicious-client population.

        Attacker identities are used exclusively
        by the poisoning simulator and evaluation.

        They are not supplied to the trust manager
        or the federated aggregation algorithm.
        """

        rng = np.random.default_rng(
            self.config.seed + 1000
        )

        count = int(
            round(
                self.config.num_clients
                * self.config.malicious_fraction
            )
        )

        selected = rng.choice(
            self.config.num_clients,
            size=count,
            replace=False,
        )

        self.malicious_clients = {
            int(client_id)
            for client_id in selected
        }

        return self.malicious_clients.copy()

    def update_trust_observations(
        self,
        probabilities: Dict[int, float],
        timestamp: float,
    ) -> None:
        """
        Update TOSA trust values using externally
        supplied malicious-behavior probabilities.

        This method deliberately does not use
        ground-truth attacker identities.
        """

        for client_id, probability in (
            probabilities.items()
        ):

            self.trust_manager.update_from_probability(
                uav_id=client_id,
                malicious_probability=probability,
                timestamp=timestamp,
            )

    def select_round_clients(
        self,
        round_index: int,
    ) -> List[int]:
        """
        Select clients for one federated round.

        AURORA:
            Apply TOSA trust eligibility.

        Baselines:
            Select from all registered clients,
            without AURORA trust filtering.

        This preserves the distinction between
        the proposed method and aggregation-only
        baselines.
        """

        if self.config.aggregation_method == "aurora":

            eligible = (
                self.trust_manager.get_eligible_clients()
            )

        else:

            eligible = list(
                range(
                    self.config.num_clients
                )
            )

        return self.partitioner.select_clients(
            round_index=round_index,
            eligible_clients=eligible,
        )

    def _save_json(
        self,
        filename: str,
        payload,
    ) -> None:
        """Write JSON experiment output."""

        path = (
            self.output_dir / filename
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                default=float,
            )

    def _save_history_csv(
        self,
    ) -> None:
        """Save per-round evaluation metrics."""

        if not self.history:
            return

        path = (
            self.output_dir
            / "round_metrics.csv"
        )

        fieldnames = list(
            self.history[0].keys()
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                self.history
            )

    def _save_checkpoint(
        self,
    ) -> None:
        """Save final model parameters."""

        if self.model is None:
            return

        torch.save(
            {
                "model_state_dict": (
                    self.model.state_dict()
                ),
                "config": asdict(
                    self.config
                ),
            },
            self.output_dir
            / "final_model.pt",
        )

    def run(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        trust_observations: Optional[
            Dict[int, Dict[int, float]]
        ] = None,
    ) -> Dict:
        """
        Execute the complete FL experiment.

        Parameters
        ----------
        train_features, train_labels:
            Training dataset, already preprocessed.

        val_features, val_labels:
            Independent validation dataset.

        test_features, test_labels:
            Independent held-out test dataset.

        trust_observations:
            Optional mapping:
                round_index -> {
                    client_id: malicious_probability
                }

            The probabilities must originate from
            an independent behavioral detector
            or recorded telemetry evaluation.

            Missing observations do not change
            the existing trust value.

        Returns
        -------
        Dictionary containing final metrics,
        malicious-client assignments, and logs.
        """

        train_features, train_labels = (
            validate_dataset(
                train_features,
                train_labels,
            )
        )

        val_features, val_labels = (
            validate_dataset(
                val_features,
                val_labels,
            )
        )

        test_features, test_labels = (
            validate_dataset(
                test_features,
                test_labels,
            )
        )

        input_dim = train_features.shape[1]

        if (
            val_features.shape[1] != input_dim
            or test_features.shape[1] != input_dim
        ):
            raise ValueError(
                "Train, validation, and test "
                "feature dimensions must match."
            )

        unique_labels = np.unique(
            train_labels
        )

        if not np.array_equal(
            unique_labels,
            np.arange(len(unique_labels)),
        ):
            raise ValueError(
                "Training labels must be encoded "
                "as consecutive integers 0..C-1."
            )

        num_classes = len(
            unique_labels
        )

        if num_classes < 2:
            raise ValueError(
                "At least two classes are required."
            )

        for name, labels in (
            ("validation", val_labels),
            ("test", test_labels),
        ):

            if np.any(labels >= num_classes):
                raise ValueError(
                    f"{name} labels contain "
                    "classes not present in training."
                )

        self.history.clear()

        self.initialize_clients()

        self.partition = (
            self.partitioner.dirichlet_partition(
                train_labels
            )
        )

        self.assign_malicious_clients()

        self.model = TabularFLModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self._save_json(
            "experiment_config.json",
            asdict(self.config),
        )

        self._save_json(
            "malicious_clients.json",
            sorted(self.malicious_clients),
        )

        global_parameters = (
            get_model_parameters(
                self.model
            )
        )

        observations = (
            trust_observations or {}
        )

        total_malicious_selected = 0
        total_malicious_excluded = 0

        for round_index in range(
            1,
            self.config.rounds + 1,
        ):

            if round_index in observations:

                self.update_trust_observations(
                    probabilities=observations[
                        round_index
                    ],
                    timestamp=float(
                        round_index
                    ),
                )

            selected_clients = (
                self.select_round_clients(
                    round_index
                )
            )

            client_updates = []

            local_losses = []

            for client_id in selected_clients:

                client_x, client_y = (
                    self.partition.get_client_data(
                        client_id,
                        train_features,
                        train_labels,
                    )
                )

                local_parameters, local_loss = (
                    train_local_model(
                        global_parameters=(
                            global_parameters
                        ),
                        input_dim=input_dim,
                        num_classes=num_classes,
                        hidden_dim=(
                            self.config.hidden_dim
                        ),
                        features=client_x,
                        labels=client_y,
                        epochs=(
                            self.config.local_epochs
                        ),
                        batch_size=(
                            self.config.batch_size
                        ),
                        learning_rate=(
                            self.config.learning_rate
                        ),
                        device=str(self.device),
                        seed=(
                            self.config.seed
                            + round_index * 10000
                            + client_id
                        ),
                    )
                )

                local_losses.append(
                    local_loss
                )

                if client_id in self.malicious_clients:

                    attack_rng = (
                        np.random.default_rng(
                            [
                                self.config.seed,
                                round_index,
                                client_id,
                                2026,
                            ]
                        )
                    )

                    local_parameters = (
                        apply_model_update_poisoning(
                            global_parameters=(
                                global_parameters
                            ),
                            local_parameters=(
                                local_parameters
                            ),
                            strength=(
                                self.config.poisoning_strength
                            ),
                            rng=attack_rng,
                        )
                    )

                trust_value = (
                    self.trust_manager.get_trust(
                        client_id
                    )
                )

                client_updates.append(
                    ClientUpdate(
                        client_id=client_id,
                        parameters=local_parameters,
                        num_samples=len(client_y),
                        trust=trust_value,
                    )
                )

            aggregation_result = (
                self.aggregator.aggregate(
                    client_updates,
                    method=(
                        self.config.aggregation_method
                    ),
                )
            )

            global_parameters = (
                aggregation_result.parameters
            )

            set_model_parameters(
                self.model,
                global_parameters,
            )

            validation_metrics = (
                evaluate_model(
                    model=self.model,
                    features=val_features,
                    labels=val_labels,
                    device=str(self.device),
                )
            )

            selected_malicious = (
                set(selected_clients)
                & self.malicious_clients
            )

            excluded_malicious = (
                selected_malicious
                & set(
                    aggregation_result.excluded_clients
                )
            )

            # This metric measures exclusion among
            # selected malicious participants.
            # It is not equivalent to all possible
            # definitions of poisoning resistance.

            total_malicious_selected += len(
                selected_malicious
            )

            total_malicious_excluded += len(
                excluded_malicious
            )

            round_exclusion_rate = (
                100.0
                * len(excluded_malicious)
                / len(selected_malicious)
                if selected_malicious
                else None
            )

            record = {
                "round": round_index,
                "selected_clients": len(
                    selected_clients
                ),
                "aggregated_clients": len(
                    aggregation_result.participating_clients
                ),
                "malicious_selected": len(
                    selected_malicious
                ),
                "malicious_excluded": len(
                    excluded_malicious
                ),
                "malicious_exclusion_rate": (
                    round_exclusion_rate
                ),
                "local_loss": float(
                    np.mean(local_losses)
                ),
                "validation_accuracy": (
                    validation_metrics["accuracy"]
                ),
                "validation_loss": (
                    validation_metrics["loss"]
                ),
                "aggregation_time_ms": (
                    aggregation_result.aggregation_time_ms
                ),
                "trust_stability": (
                    self.trust_manager.trust_stability()
                ),
            }

            self.history.append(
                record
            )

            print(
                f"Round {round_index:03d}/"
                f"{self.config.rounds:03d} | "
                f"Selected={len(selected_clients):02d} | "
                f"Val-ACC="
                f"{validation_metrics['accuracy']:.2f}% | "
                f"Loss="
                f"{validation_metrics['loss']:.4f}"
            )

        final_test_metrics = (
            evaluate_model(
                model=self.model,
                features=test_features,
                labels=test_labels,
                device=str(self.device),
            )
        )

        malicious_exclusion_rate = (
            100.0
            * total_malicious_excluded
            / total_malicious_selected
            if total_malicious_selected > 0
            else None
        )

        summary = {
            "aggregation_method": (
                self.config.aggregation_method
            ),
            "rounds": self.config.rounds,
            "num_clients": (
                self.config.num_clients
            ),
            "clients_per_round": (
                self.config.clients_per_round
            ),
            "malicious_clients": sorted(
                self.malicious_clients
            ),
            "final_test_accuracy": (
                final_test_metrics["accuracy"]
            ),
            "final_test_loss": (
                final_test_metrics["loss"]
            ),
            "malicious_exclusion_rate": (
                malicious_exclusion_rate
            ),
            "trust_stability": (
                self.trust_manager.trust_stability()
            ),
            "history": self.history,
        }

        self._save_json(
            "experiment_summary.json",
            summary,
        )

        self._save_history_csv()

        self._save_checkpoint()

        return summary


def create_synthetic_dataset(
    num_samples: int = 12000,
    num_features: int = 20,
    num_classes: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset
    for testing the FL workflow.

    This dataset is not a substitute for the
    public datasets used in the manuscript.
    """

    rng = np.random.default_rng(
        seed
    )

    features = rng.normal(
        size=(
            num_samples,
            num_features,
        )
    ).astype(
        np.float32
    )

    weights = rng.normal(
        size=(
            num_features,
            num_classes,
        )
    )

    logits = (
        features @ weights
        + rng.normal(
            scale=0.5,
            size=(
                num_samples,
                num_classes,
            ),
        )
    )

    labels = np.argmax(
        logits,
        axis=1,
    ).astype(
        np.int64
    )

    return features, labels


if __name__ == "__main__":

    print(
        "AURORA Federated Learning Demonstration"
    )

    print("=" * 70)

    config = FLConfig(
        num_clients=50,
        clients_per_round=20,
        rounds=3,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=64,
        dirichlet_alpha=0.5,
        min_samples_per_client=10,
        malicious_fraction=0.20,
        poisoning_strength=0.20,
        trust_threshold=0.50,
        trust_alpha=0.70,
        aggregation_method="aurora",
        seed=42,
        device="cpu",
        output_dir="outputs/fl/demo",
    )

    features, labels = (
        create_synthetic_dataset(
            num_samples=12000,
            num_features=20,
            num_classes=4,
            seed=42,
        )
    )

    rng = np.random.default_rng(
        42
    )

    indices = rng.permutation(
        len(labels)
    )

    train_end = int(
        0.70 * len(indices)
    )

    val_end = int(
        0.85 * len(indices)
    )

    train_indices = indices[
        :train_end
    ]

    val_indices = indices[
        train_end:val_end
    ]

    test_indices = indices[
        val_end:
    ]

    trainer = AURORAFederatedTrainer(
        config
    )

    summary = trainer.run(
        train_features=features[
            train_indices
        ],
        train_labels=labels[
            train_indices
        ],
        val_features=features[
            val_indices
        ],
        val_labels=labels[
            val_indices
        ],
        test_features=features[
            test_indices
        ],
        test_labels=labels[
            test_indices
        ],
    )

    print(
        "\nFinal Test Accuracy:",
        f"{summary['final_test_accuracy']:.2f}%"
    )

    print(
        "Trust Stability:",
        f"{summary['trust_stability']:.2f}%"
    )

    print(
        "Malicious Exclusion Rate:",
        summary["malicious_exclusion_rate"],
    )

    print(
        "\nExperimental outputs saved to:",
        config.output_dir,
    )
