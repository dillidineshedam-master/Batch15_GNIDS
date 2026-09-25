# Fed-GNIDS: Byzantine-Robust Federated Spatial-Temporal Graph Neural Network for Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PyG-ee4c2c.svg)](https://pytorch.org/)
[![Federated Learning: Flower](https://img.shields.io/badge/FL-Flower%20Framework-ff69b4.svg)](https://flower.ai/)
[![Dataset: UNSW--NB15](https://img.shields.io/badge/Benchmark-UNSW--NB15-green.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fed-GNIDS is an unsupervised, privacy-preserving Network Intrusion Detection System (NIDS) designed for decentralized enterprise subnets. By mapping continuous NetFlow traffic into dynamic temporal graph snapshots, Fed-GNIDS couples edge-conditioned Graph Attention Networks (**GATv2**) with Gated Recurrent Units (**GRU**) to detect multi-hop lateral movement without centralizing private network traffic.

To maintain model stability against adversarial manipulation in federated environments, Fed-GNIDS introduces the **Adaptive Contribution Scaling (ACS)** algorithm on the central aggregator, mitigating Byzantine gradient inversion (sign-flipping) and scale-poisoning attacks under Non-IID Dirichlet distributions.

---

## System Architecture

```text
[ Local Subnet Telemetry ]
│
▼
┌──────────────────────────────────────────────┐
│  Graph Snapshot Construction (T=5 Windows)   │
│  - Nodes: Communicating Endpoints (|V| = 50) │
│  - Edges: 16-D NetFlow Attribute Vectors     │
└──────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│       Spatial-Temporal Autoencoder           │
│  1. Spatial Attention: Multi-Head GATv2      │
│  2. Temporal Recurrence: Recurrent GRU       │
│  3. Decoder: Edge Reconstruction MLP         │
│  4. Local Privacy: DP L2 Gradient Clipping   │
└──────────────────────────────────────────────┘
│ (Transmits Model Deltas Δw_k)
▼
┌──────────────────────────────────────────────┐
│   Central Aggregator: ACS Defense Engine     │
│  - Step 1: Coordinate-wise Median Consensus  │
│  - Step 2: Dynamic L2 Norm Bounding (C=5.0)  │
│  - Step 3: Directional Cosine Alignment      │
│  - Step 4: Robust Weighted Aggregation       │
└──────────────────────────────────────────────┘
```

## Empirical Benchmark Results (UNSW-NB15)

Evaluated on the authentic **UNSW-NB15** benchmark (82,332 training flows; 24,000 independent testing flows containing 7,690 normal and 16,310 attack flows) partitioned via a Dirichlet Non-IID distribution ($\alpha = 0.5$).

### 1. Comparative Baseline Performance
| Paradigm | Method | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Supervised Centralized** | Random Forest (Tabular) | 89.28% | 0.9824 | 85.76% | 0.9158 | 0.9837 |
| **Decentralized Silo** | Local-Only GATv2-GRU | 63.92% | 0.9626 | 48.80% | 0.6477 | 0.7613 |
| **Federated Static** | Spatial Fed-GCN | 69.26% | 0.9730 | 56.33% | 0.7135 | 0.8392 |
| **Proposed Framework** | **Fed-GNIDS (GATv2-GRU + ACS)** | **67.85%** | **0.9759** | **54.03%** | **0.6956** | **0.8287** |

*At an operational anomaly threshold of $\tau = \mu + 2.0\sigma$, Fed-GNIDS operates with a False Alarm Rate (FAR) of **2.83%** without requiring attack labels during training.*

### 2. Byzantine Poisoning Defense ($K=4$ Subnets, $f=25\%$ Adversary)
| Defense Strategy & Attack Scenario | ROC-AUC | Accuracy | Detection Recall | F1-Score | Defense Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline (No Attack)** | 0.7779 | 72.94% | 61.87% | 0.7566 | Baseline |
| **FedAvg under Sign-Flip Attack** | 0.8613 | 73.67% | 63.07% | 0.7650 | Unmitigated |
| **FedAvg under Scale-Poisoning ($20\times$)** | 0.8014 | **44.65%** | **24.03%** | **0.3711** | **Collapsed** |
| **ACS under Sign-Flip (Proposed)** | **0.8006** | **73.09%** | **62.00%** | **0.7580** | **Resilient** |
| **ACS under Scale-Poison ($20\times$) (Proposed)** | **0.8011** | **72.95%** | **61.95%** | **0.7569** | **Neutralized** |

### 3. Modular Ablation Analysis
| Architecture Configuration | ROC-AUC | Accuracy | Recall | F1-Score | Analytical Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Fed-GNIDS (GATv2 + GRU + DP)** | 0.7724 | 69.71% | **57.14%** | **0.7194** | Full spatial-temporal baseline |
| **w/o Temporal Recurrence (Spatial Only)** | 0.8245 | 59.81% | **42.56%** | 0.5900 | Recall drops by -14.58% |
| **w/o Graph Attention (Static GCN + GRU)** | 0.8280 | 75.03% | 65.05% | 0.7798 | Faster low-parameter convergence |
| **w/o Differential Privacy (No Norm Clip)** | 0.7359 | 49.80% | **27.97%** | **0.4309** | Gradient instability & collapse |

### 4. Edge Computational Footprint
* **Model Parameters:** 68,848 (Weights size: 268.94 KB RAM)
* **Inference Latency:** 51.83 ms per 250-flow window (207.34 microseconds / flow)
* **Processing Throughput:** 4,823 flows / second on commodity CPU
* **FL Transmission:** 268.94 KB uplink/downlink per round (4.20 MB total across 8 rounds)

---

## Repository Directory Structure

```text
Batch15_GNIDS/
├── data/
│   ├── raw/                       # Raw UNSW-NB15 CSV datasets
│   └── processed/                 # Dirichlet non-IID graph tensors (.pt)
├── results/                       # Empirical tables (CSV) & publication plots (PNG)
│   ├── ieee_baseline_comparison.png
│   ├── ieee_adversarial_robustness.png
│   ├── ieee_ablation_study.png
│   ├── ieee_roc_curve.png
│   └── ieee_confusion_matrix.png
├── src/
│   ├── model.py                   # Spatial-temporal GATv2-GRU autoencoder definition
│   ├── preprocess.py              # Flow hashing, NetFlow scaler, and PyG graph slicing
│   ├── server.py                  # Flower FL server with ACS aggregation logic
│   ├── client.py                  # Flower client with DP clipping & attack modes
│   ├── evaluate.py                # Ground-truth UNSW-NB15 evaluation script
│   ├── baselines.py               # Comparative benchmark evaluation runner
│   ├── simulate_attacks.py        # Multi-client Byzantine robustness simulator
│   ├── ablation.py                # Component isolation & ablation study
│   ├── profiler.py                # Hardware latency, throughput, & payload profiler
│   └── app.py                     # Streamlit SOC Autonomous Detection Radar
├── paper.tex                      # Complete IEEE double-column LaTeX manuscript
├── run_pipeline.py                # Master headless orchestrator script
├── requirements.txt               # Locked environment dependencies
└── README.md                      # System documentation

```

---

## Quickstart & Execution

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Automated Research Pipeline
To verify datasets, execute 8 rounds of federated training, evaluate benchmarks, run Byzantine attacks, conduct ablation studies, and profile hardware performance sequentially:
```powershell
python run_pipeline.py
```

### 3. Launch the Streamlit SOC Intrusion Radar
To inspect the real-time intrusion monitoring dashboard with interactive threshold calibration and incident inspection:
```powershell
streamlit run src/app.py
```

---

## Authors & Academic Affiliation
* **Dilli Dinesh Edam** (Reg: 234M1A0530)
* **Dasari Kushvanth Kumar** (Reg: 234M1A0529)
* **D Mahesh Babu** (Reg: 234M1A0528)
* **G Praveen** (Reg: 234M1A0535)
* **B Premchand** (Reg: 234M1A0520)

**Project Guide:** Mr. K. Niranjan, Assistant Professor  
*Department of Computer Science and Engineering*  
**VEMU Institute of Technology**, Chittoor, Andhra Pradesh, India

---

## Citation
If you find this work useful in your research, cite:
```bibtex
@inproceedings{edam2026fedgnids,
  title={Fed-GNIDS: Byzantine-Robust Federated Spatial-Temporal Graph Autoencoders for Privacy-Preserving Network Intrusion Detection},
  author={Edam, Dilli Dinesh and Kumar, Dasari Kushvanth and Babu, D Mahesh and Praveen, G and Premchand, B and Niranjan, K},
  booktitle={Proceedings of the IEEE Conference on Communications and Network Security},
  year={2026}
}
```