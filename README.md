# Privacy-Preserving Federated Learning for Healthcare (`healthcare-fl`)

An empirical research implementation and benchmark of **Privacy-Preserving Federated Learning (FL)** applied to distributed medical radiograph classification on the **NIH ChestX-ray14** dataset, incorporating **Differential Privacy (DP-SGD)** with a **Dynamic Privacy Budget Guard**.

---

## 1. Project Overview

Patient confidentiality regulations (HIPAA, GDPR) strictly restrict the centralization of raw diagnostic imaging across medical institutions. Federated Learning (FedAvg) allows multiple institutions to collaboratively train shared diagnostic neural networks without exchanging raw patient records. However, vanilla FedAvg provides no formal differential privacy guarantees against gradient-based information leakage.

This project implements and empirically audits:
1. **5 Simulated Hospital Clients:** Constructed as synthetic, patient-isolated partitions using frontal chest radiographs from the public NIH ChestX-ray14 dataset.
2. **Binary Clinical Pathology Task:** Detection of **Pneumonia** ($y = 1.0$) against verified clean controls ($y = 0.0$, patients with "No Finding" who never had a pneumonia diagnosis in the dataset).
3. **Opacus Differential Privacy (DP-SGD):** Incorporating Poisson subsampling ($q = 0.3333$), per-sample gradient clipping ($C = 1.0$), Gaussian noise injection ($\sigma = 1.0$), persistent Rényi/PRV privacy accounting, and Adam momentum buffer clearing on parameter synchronization.
4. **Dynamic Privacy Budget Guard:** A pre-round projection mechanism enforcing a cumulative budget limit of $\epsilon \le 10.0$ at target $\delta = 10^{-5}$, halting training dynamically before any round whose projected privacy expenditure would exceed the ceiling.
5. **Multi-Baseline Benchmark:** Comparing Centralized ML, Vanilla FedAvg (no DP), FedAvg + DP Baseline (stale optimizer state), FedAvg + DP Corrected (reset optimizer state), and FedAvg + DP with 3 Local Epochs.

---

## 2. Directory Structure

```text
healthcare-fl/
├── config.yaml                         # Project configuration and hyperparameters
├── clients/
│   ├── hospital_client.py              # Base non-private hospital client
│   └── hospital_dp_client.py           # Opacus DP-SGD client with Adam reset
├── models/
│   └── cnn.py                          # SmallChestXrayCNN with GroupNorm (num_groups=4)
├── server/
│   └── federated_server.py             # Flower-based weighted FedAvg server
├── experiments/
│   ├── centralized_baseline.py         # Centralized training baseline
│   ├── fedavg_baseline.py              # Vanilla FedAvg baseline runner
│   ├── fedavg_dp_baseline.py           # Original DP baseline runner
│   ├── fedavg_dp_corrected.py          # Corrected DP baseline runner
│   └── fedavg_dp_3epoch.py             # 3-local-epoch DP runner with dynamic budget guard
├── tests/
│   ├── test_partitioning.py            # Cohort isolation & disjointness unit tests
│   ├── test_cnn_data_loading.py        # Data loading & DP compliance unit tests
│   ├── test_fedavg.py                  # Federated aggregation & parameter round-trip tests
│   └── test_fedavg_dp_3epoch.py        # Dedicated 8-test suite for budget guard & accounting
├── results/
│   ├── centralized_baseline/           # Centralized model checkpoint & history
│   ├── fedavg_baseline/                # Vanilla FedAvg model checkpoint & history
│   ├── fedavg_dp_baseline/             # Original DP baseline artifacts
│   ├── fedavg_dp_corrected/            # Corrected DP baseline artifacts
│   ├── fedavg_dp_3epoch/               # 3-epoch DP baseline artifacts
│   └── submission/                     # Final verified submission package
│       ├── plots/                      # Publication-quality benchmark figures
│       ├── IMPLEMENTATION_GUIDE.md     # Full architectural and mathematical guide
│       ├── RESULTS_SUMMARY.md          # Complete experimental metrics & comparisons
│       ├── KNOWN_LIMITATIONS.md        # Methodological & clinical limitations
│       ├── TEST_REPORT.md              # 26-test verification audit
│       ├── REPRODUCIBILITY.md          # Reproduction commands & SHA256 hashes
│       └── PROGRESS.md                 # Sequential phase execution log
└── requirements.txt                    # Pinned Python package dependencies
```

---

## 3. Core Experimental Results

All experiments evaluated the same binary classification task on identical patient cohorts with a held-out test split of 30 unseen patients (15 Pneumonia, 15 Control).

> [!IMPORTANT]
> **Checkpoint Selection Protocol:** In all federated experiments, global checkpoints were selected strictly by highest mean client validation F1 at fixed threshold $0.50$, with ties broken in favor of the earliest round. Secondary metrics (loss or ROC-AUC) were not permitted to break ties. Held-out test evaluation was executed strictly once on the selected checkpoint.

| Baseline / Model | Training Paradigm | Local Epochs ($E$) | Completed Rounds | Cumulative Steps | Realized Privacy ($\delta=10^{-5}$) | Selected Round | Selected Checkpoint Val ROC-AUC | Latest Completed Round Val ROC-AUC | Held-Out Test Accuracy | Held-Out Test F1 ($\tau=0.5$) | Held-Out Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Centralized ML | 5 (Global) | N/A | 15 (Global) | None ($\infty$) | Epoch 2 | 0.8229 | 0.8125 | **76.67%** | **0.7407** | **0.8400** |
| **Vanilla FedAvg** | 5 Simulated Clients | 1 | 5 | 15 | None ($\infty$) | Round 3 | 0.7056 | 0.7389 | **56.67%** | **0.6667** | **0.7556** |
| **FedAvg + DP Baseline** | 5 Simulated Clients | 1 | 5 | 15 | $\epsilon = 9.1137$ | Round 3* | 0.6333 | 0.6333 | 50.00% | 0.0000 | 0.5022 |
| **FedAvg + DP Corrected** | 5 Simulated Clients | 1 | 5 | 15 | $\epsilon = 9.1137$ | Round 1 | 0.6222 | 0.5722 | 50.00% | 0.0000 | 0.4400 |
| **FedAvg + DP ($E=3$)** | 5 Simulated Clients | 3 | **2 (Budget Stop)** | 18 | **$\mathbf{\epsilon = 9.9496}$** | **Round 1** | **0.6167** | **0.7278** | 50.00% | 0.0000 | **0.5067** |

*\*Note: In the original DP baseline, Round 3 was selected under an earlier loss tie-breaker; under the strict earliest-tie rule (implemented in the corrected and 3-epoch experiments), Round 1 is selected when all rounds achieve F1 = 0.0.*

---

## 4. Key Scientific & Technical Findings

1. **Multi-Epoch Training Trajectory vs. Checkpoint Selection:**
   - In `fedavg_dp_3epoch`, increasing local epochs from $E=1$ (3 steps/round) to $E=3$ (9 steps/round) increased mean client validation ROC-AUC from $0.6167$ in Round 1 to **$0.7278$ in Round 2**.
   - However, because both Round 1 and Round 2 produced mean validation F1 = 0.0000 at the fixed decision threshold $0.50$, the earliest-tie selection rule retained **Round 1**. Consequently, the checkpoint evaluated on the held-out test set was Round 1 (validation ROC-AUC 0.6167, test ROC-AUC 0.5067), not Round 2.
2. **Behavior of Default Decision Threshold (0.50):**
   - Across all three differential privacy experiments, all predicted probabilities on both the client validation sets and the held-out test set remained below 0.50, causing all binary predictions to be negative ($\text{FN}=15, \text{TN}=15, \text{TP}=0, \text{FP}=0$, test accuracy 50.00%, F1 = 0.0000).
   - In contrast, threshold-independent ranking metrics (validation ROC-AUC 0.6167–0.7278) show non-random ordering. In practical settings, decision thresholds require validation-based calibration rather than an uncalibrated default threshold of 0.50.
3. **Dynamic Privacy Budget Gating:**
   - The non-mutating PRV accountant projection calculated that executing Round 3 would expend cumulative $\epsilon = 12.1878 > 10.0$ at $\delta = 10^{-5}$. The server cleanly halted training before Round 3, ensuring final realized privacy was capped at $\epsilon = 9.9496$.
4. **Privacy Scope & Threat Model:**
   - Differential privacy guarantees that the presence or absence of any single patient's record boundedly alters the output distribution of the model parameters. An $\epsilon = 9.9496$ at $\delta = 10^{-5}$ provides a formal theoretical bound, but whether this level of epsilon constitutes adequate protection depends on the specific threat model, background knowledge, and clinical context.

---

## 5. Non-Destructive Verification Commands

All checks use the existing virtual environment without re-running training or modifying artifacts:

```powershell
# 1. Run core test suite (26 tests, 0 failures, strictly zero test set access)
.\.venv\Scripts\pytest.exe tests\test_fedavg_dp_3epoch.py tests\test_partitioning.py tests\test_cnn_data_loading.py tests\test_fedavg.py -v

# 2. Verify all 54 artifact hashes (42 protected baseline + 12 3-epoch artifacts)
.\.venv\Scripts\python.exe -c "
import hashlib, json; from pathlib import Path
b1 = json.loads(Path('results/baseline_integrity_hashes.json').read_text())
b2 = json.loads(Path('results/fedavg_dp_corrected_integrity_hashes.json').read_text())
b3 = json.loads(Path('results/fedavg_dp_3epoch_integrity_hashes.json').read_text())
all_54 = {**b1, **{f'fedavg_dp_baseline/{k}': v for k, v in b2.items() if 'corrected' not in k}, **{f'fedavg_dp_corrected/{k}': v for k, v in b2.items()}, **{f'fedavg_dp_3epoch/{k}': v for k, v in b3.items()}}
all_54['fedavg_dp_baseline/fedavg_dp_global_model.pt'] = 'e5de6bd5684ed4652624efde55921370f2dece35be04bbcb8f29beeddb5ad759'
all_54['fedavg_dp_baseline/fedavg_dp_training_history.json'] = '2ee7871355a9e987877b0a1d38c4f91722d88b71eff151659a6c4e99e953b2b9'
mismatches = [f for f, h in all_54.items() if hashlib.sha256(Path('results/' + f).read_bytes()).hexdigest() != h]
print('Verification Result:', 'ALL 54 ARTIFACTS VERIFIED IDENTICAL' if not mismatches else mismatches)
"
```

For complete documentation:
- [`results/submission/IMPLEMENTATION_GUIDE.md`](file:///c:/Users/Vaishnavi%20Sinha/OneDrive/Desktop/SRM/Project/healthcare-fl/results/submission/IMPLEMENTATION_GUIDE.md)
- [`results/submission/RESULTS_SUMMARY.md`](file:///c:/Users/Vaishnavi%20Sinha/OneDrive/Desktop/SRM/Project/healthcare-fl/results/submission/RESULTS_SUMMARY.md)
- [`results/submission/KNOWN_LIMITATIONS.md`](file:///c:/Users/Vaishnavi%20Sinha/OneDrive/Desktop/SRM/Project/healthcare-fl/results/submission/KNOWN_LIMITATIONS.md)
- [`results/submission/TEST_REPORT.md`](file:///c:/Users/Vaishnavi%20Sinha/OneDrive/Desktop/SRM/Project/healthcare-fl/results/submission/TEST_REPORT.md)
- [`results/submission/REPRODUCIBILITY.md`](file:///c:/Users/Vaishnavi%20Sinha/OneDrive/Desktop/SRM/Project/healthcare-fl/results/submission/REPRODUCIBILITY.md)
- [`results/submission/PROGRESS.md`](file:///c:/Users/Vaishnavi%20Sinha/OneDrive/Desktop/SRM/Project/healthcare-fl/results/submission/PROGRESS.md)
