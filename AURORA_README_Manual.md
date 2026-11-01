# AURORA: Operations Intelligence for Risk-Aware Federated Learning and Adaptive Decision-Making in UAV-Assisted Transportation Systems

**AURORA** integrates programmable network telemetry, behavioral trust assessment, risk-aware federated learning, and adaptive security responses for UAV-assisted transportation systems. We organize the framework into **TOSA** (assessment), **RFLD** (learning protection), and **ACRO** (response orchestration).

> **Release scope:** The available reference implementation supports package tests, synthetic comparisons, sensitivity checks, and CSV benchmarking. The expanded 38-module architecture and complete UAV/P4/FL workflow are **not yet verified as an integrated executable release**. Results below are **reported in the manuscript**, not independently reproduced by this repository.

## 1. Repository architecture

```text
AURORA/
├── aurora/
│   ├── cli.py                         # Reference CLI
│   ├── config.py                      # Runtime configuration
│   ├── tosa/
│   │   ├── telemetry.py               # Telemetry and feature handling
│   │   └── trust.py                   # Behavioral trust and risk scores
│   ├── rfld/
│   │   ├── aggregation.py             # Trust-aware / baseline aggregation
│   │   ├── partition.py               # Non-IID client data partitioning
│   │   └── learning.py                # Federated training coordination
│   ├── acro/
│   │   ├── policy.py                  # Risk-based mitigation selection
│   │   └── enforcement.py             # Network-action interface
│   ├── models/
│   │   ├── classification.py          # RF / boosting classifiers
│   │   ├── ensemble.py                # Hybrid score fusion
│   │   └── gru.py                     # Temporal classifier
│   ├── experiments/
│   │   ├── scenarios.py               # Attack scenario definitions
│   │   ├── traffic.py                 # Traffic workload generation
│   │   ├── dataset_mapping.py         # Dataset-to-traffic mapping
│   │   ├── mininet_wifi_topology.py   # UAV wireless emulation
│   │   └── ...                        # Expanded experiment/report modules
│   ├── evaluation/
│   │   ├── metrics.py                 # Evaluation indicators
│   │   ├── statistics.py              # Multi-run statistics
│   │   ├── comparison.py              # Baseline comparisons
│   │   ├── sensitivity.py             # Parameter sweeps
│   │   └── ablation.py                # Component-removal experiments
│   └── utils/                         # Logging / reproducibility / reports
├── p4/                                # P4 telemetry and compile script
├── configs/                            # Experiment configuration
├── tests/                              # Unit / integration tests
├── docs/                               # Manuscript-to-code mapping
├── results/                            # Reference CLI outputs
├── outputs/                            # Expanded experiment outputs
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

**Original flat-source mapping:** `tosa_telemetry.py` → `tosa/telemetry.py`; `tosa_trust.py` → `tosa/trust.py`; `rfld_aggregation.py`, `rfld_partition.py`, `rfld_learning.py` → `rfld/`; `acro_policy.py`, `acro_enforcement.py` → `acro/`; `aurora_models_classification.py`, `aurora_models_ensemble.py`, `aurora_models_gru.py` → `models/`; `Scenarios.py`, `traffic.py`, `dataset_mapping.py`, `mininet_wifi_topology.py`, `experiments_attack_injection.py` → `experiments/`; `evaluation_metrics.py`, `evaluation_comparison.py`, `evaluation_sensitivity.py`, `statistics.py`, `ablation.py` → `evaluation/`; `aurora_utils_logging.py`, `aurora_utils_reproducibility.py`, `monitoring.py`, `aurora_config.py` → the corresponding utilities, monitoring, and configuration modules. Retain `aurora_telemetry_P4.p4` and `compile.sh` under `p4/`. **Moving files requires updating imports and entry points.**

## 2. Installation — reference implementation

**Prerequisites:** Python 3.10+, Git. Use Linux for optional Mininet-WiFi/P4 networking experiments.

**Step 1 — Clone and enter the repository**

```bash
git clone https://github.com/<OWNER>/AURORA.git
cd AURORA
```

**Step 2 — Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

**Step 3 — Install the package and test dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

**Step 4 — Verify the installation**

```bash
python -m pytest -q
```

**Step 5 — Install optional components as needed**

```bash
python -m pip install -e '.[boosting]'
python -m pip install -e '.[deep]'
python -m pip install -e '.[network]'
```

The optional extras above belong to the reference package. Check `pyproject.toml` before using them. The reference GRU backend uses optional TensorFlow; the expanded design specifies PyTorch. **Select and document one backend before combining implementations.**

## 3. Execute the available reference workflows

Run the following commands from the repository root after activating `.venv`.

**Step 6 — Export scenario definitions**

```bash
aurora scenarios --out results/scenarios.json
```

**Step 7 — Run a synthetic baseline comparison**

```bash
aurora compare --synthetic-samples 4000 --seeds 42 --out results/comparison.json
```

**Step 8 — Run a synthetic sensitivity check**

```bash
aurora sensitivity --synthetic-samples 4000 --seeds 42 --out results/sensitivity.json
```

**Step 9 — Evaluate a labeled CSV dataset**

```bash
aurora benchmark \
  --csv /path/to/dataset.csv \
  --label label \
  --group capture_id \
  --models rf \
  --out results/benchmark.json
```

Replace the CSV path, label column, and group column with real dataset fields. Omit `--group` only if no suitable grouping field exists; document possible leakage limitations. The CLI accepts the reference YAML configuration using `--config configs/default.yaml`.

> **Interpretation:** Steps 6–8 use synthetic inputs; they verify engineering workflows, **not** the manuscript's benchmark or UAV-emulation results.

## 4. Full experimental workflow — integration checklist

The following numbered sequence describes the **intended** manuscript-aligned workflow. It requires integration and validation of the expanded modules before end-to-end execution.

| Step | Action | Primary files | Expected output |
|---|---|---|---|
| **10** | Acquire and verify CICIoT2023, CICDDoS2019, TON_IoT, Edge-IIoTset | `dataset_mapping.py`, dataset loaders | Dataset manifest and checksums |
| **11** | Preprocess, encode, and split data (70/15/15) | Model and data utilities | Train/validation/test partitions |
| **12** | Train RF, XGBoost, LightGBM, GRU, and ensemble | `models/` | Model outputs and detection metrics |
| **13** | Compile the P4 telemetry program | `p4/compile.sh` | BMv2-compatible P4 artifacts |
| **14** | Initialize Mininet-WiFi UAV topology and legitimate traffic | `mininet_wifi_topology.py`, `traffic.py` | Network event/traffic logs |
| **15** | Execute configured attack scenarios S1–S10 | `scenarios.py`, attack-injection module | Attack and observation records |
| **16** | Derive behavioral scores and update UAV trust | `tosa/` | Trust/risk trajectories |
| **17** | Partition non-IID client data and run FL rounds | `rfld/` | Client/global model updates |
| **18** | Compare RFLD with FedAvg, Krum, Trimmed Mean | `aggregation.py`, `comparison.py` | FL accuracy, PRR, aggregation time |
| **19** | Apply ACRO policies and measure *actual* enforcement | `acro/`, monitoring | Mitigation and latency records |
| **20** | Run sensitivity, ablation, and 10-seed statistics | `evaluation/` | Per-run results and summaries |
| **21** | Trace tables/figures to measured outputs | Reporting utilities | Reproducibility package |

**Optional P4 compilation**, after installing a compatible `p4c`/BMv2 toolchain:

```bash
bash p4/compile.sh
```

Do not assume the compiled P4 example or ACRO action-selection logic establishes complete P4Runtime deployment or verified network enforcement.

## 5. Experimental configuration reported in the paper

| Parameter | Value |
|---|---:|
| UAV population (network emulation) | 20–100 |
| UAV speed | 10–20 m/s |
| UAV altitude | 50–120 m |
| Communication range | 250–500 m |
| Emulation duration | 600 s |
| FL clients | 50 |
| Clients selected per round | 20 |
| Communication rounds | 50–100 |
| Local epochs / batch size | 5 / 32 |
| Non-IID Dirichlet concentration | 0.5 |
| Malicious-client fraction | 20% |
| Model-update perturbation | 20% of original update norm |
| Trust eligibility threshold, τT | 0.50 |
| Trust-memory coefficient, α | 0.70 |
| Risk thresholds | 0.30 / 0.70 |
| Isolation condition | 3 consecutive high-risk intervals |
| Independent detection runs | 10 |

**ACRO action priority:** Isolation > Filtering > Rate limiting > Rerouting > Resource adjustment > Monitoring.

## 6. Consolidated manuscript results

**These tables summarize reported manuscript findings; they are not generated or verified by the available reference package.**

### 6.1 Detection accuracy — ten-run means (Table 5A)

| Dataset | GRU (%) | AURORA (%) | Gain (percentage points) |
|---|---:|---:|---:|
| CICIoT2023 | 96.10 | **98.60** | +2.50 |
| CICDDoS2019 | 97.10 | **99.10** | +2.00 |
| TON_IoT | 95.60 | **97.80** | +2.20 |
| Edge-IIoTset | 96.40 | **98.20** | +1.80 |

For CICIoT2023, the manuscript reports **AURORA 98.60 ± 0.25%** (95% CI: 98.42–98.78%) and **GRU 96.10 ± 0.30%** (95% CI: 95.89–96.31%).

### 6.2 Federated aggregation under poisoning (Table 6A)

| Method | FL accuracy (%) | Poisoning resistance (%) | Aggregation time (ms) |
|---|---:|---:|---:|
| FedAvg | 91.8 | 79.4 | 0.18 |
| Krum | 94.5 | 90.7 | 0.41 |
| Trimmed Mean | 95.6 | 92.8 | 0.29 |
| **AURORA-RFLD** | **98.1** | **97.6** | **0.24** |

### 6.3 Trust parameter sensitivity (Table 6B)

| Trust threshold τT | Memory α | FL accuracy (%) | PRR (%) | TS (%) |
|---:|---:|---:|---:|---:|
| 0.30 | 0.70 | 96.2 | 93.1 | 96.8 |
| 0.40 | 0.70 | 97.3 | 95.4 | 97.0 |
| **0.50** | **0.70** | **98.1** | **97.6** | **97.3** |
| 0.60 | 0.70 | 97.8 | 98.0 | 97.2 |
| 0.70 | 0.70 | 96.9 | 98.3 | 97.1 |
| 0.50 | 0.50 | 97.2 | 96.5 | 95.8 |
| 0.50 | 0.90 | 97.5 | 96.9 | 98.0 |

**Scenario results:** S3-FLPA reports **98.4% PRR**; S9-BUCA reports **97.8% PRR** (manuscript Tables 8/8A).

## 7. Outputs and manuscript mapping

| Study element | Repository component | Manuscript location |
|---|---|---|
| TOSA behavioral assessment | `tosa/`, `models/` | §4.1, Algorithm 1 |
| Trust-aware FL | `rfld/` | §4.2, Algorithm 2 |
| Adaptive mitigation | `acro/` | §4.3, Algorithm 3 |
| Classification benchmark | `models/`, `evaluation/metrics.py` | §5.2, Table 5 |
| Multi-run statistics | `evaluation/statistics.py` | Table 5A |
| Aggregation comparison | `rfld/aggregation.py`, `evaluation/comparison.py` | Table 6A |
| Trust sensitivity | `evaluation/sensitivity.py` | Table 6B |
| Attack scenarios | `experiments/` | §5.5, Tables 8/8A |
| Ablation | `evaluation/ablation.py` | §5.6, Table 9 |

Reference CLI output paths: `results/scenarios.json`, `results/comparison.json`, `results/sensitivity.json`, and `results/benchmark.json`. Expanded experiments may use `outputs/` once their reporting pipeline is integrated.

## 8. Reproducibility checks before claiming manuscript replication

1. **Datasets:** Record dataset versions, checksums, label mappings, and split indices.
2. **Models:** Save hyperparameters, trained checkpoints, preprocessing transforms, and seeds.
3. **FL:** Record client partitions, sampled participants, poisoned updates, and aggregation weights.
4. **Trust:** Use **measured TOSA predictions**, not oracle-proxy trust derived from known malicious labels.
5. **Network:** Verify real traffic injection, P4 execution, ACRO enforcement, and measured latency.
6. **Statistics:** Preserve independent per-run records, means, standard deviations, confidence intervals, and statistical tests.
7. **Traceability:** Link every reported figure and table to source experiment outputs.

**Do not treat synthetic CLI output, policy decisions without enforcement, or copied manuscript values as reproduced experimental evidence.**

## 9. Troubleshooting

| Problem | Check |
|---|---|
| `aurora: command not found` | Activate `.venv`; reinstall using `pip install -e '.[test]'` |
| Import errors after moving files | Update package imports, `__init__.py`, and entry points |
| Missing CSV column | Check `--label` and `--group` against the actual CSV headers |
| P4 compilation failure | Verify `p4c`, BMv2 target, source path, and `compile.sh` |
| Mininet-WiFi failure | Check Linux environment, dependencies, and required privileges |
| Inconsistent results | Verify data splits, seeds, configurations, and software versions |

## 10. Citation, license, and contact

**Paper:** *AURORA: Operations Intelligence for Risk-Aware Federated Learning and Adaptive Decision-Making in UAV-Assisted Transportation Systems*  
**Journal:** *Journal of Operations Intelligence* (manuscript no. 90; final bibliographic details pending)  
**Corresponding author:** Ameer El-Sayed, Faculty of Computers and Informatics, Zagazig University, Egypt  
**Email:** aegouda@fci.zu.edu.eg  
**License:** See [`LICENSE`](LICENSE) and confirm its terms before redistribution.
