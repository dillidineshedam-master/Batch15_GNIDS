# Fed-GNIDS: Byzantine-Robust Federated Spatial-Temporal Graph Neural Networks for Line-Rate Intrusion Detection at the Edge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PyG-ee4c2c.svg)](https://pytorch.org/)
[![Federated Learning: Flower](https://img.shields.io/badge/FL-Flower%20Framework-ff69b4.svg)](https://flower.ai/)
[![Dataset: UNSW--NB15](https://img.shields.io/badge/Benchmark-UNSW--NB15-green.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fed-GNIDS is a high-throughput, privacy-preserving Network Intrusion Detection System (NIDS) engineered for decentralized edge gateways and enterprise subnets. By mapping sequential NetFlow traffic into dynamic temporal graph snapshots, Fed-GNIDS couples dynamic edge-conditioned Graph Attention Networks (**GATv2**) with Gated Recurrent Units (**GRU**) to detect multi-hop lateral movement and distributed cyber threats without centralizing private network payload logs.

To counter adversarial manipulation in untrusted federated environments, the orchestrator introduces **Adaptive Contribution Scaling (ACS)**, neutralizing Byzantine gradient inversion (sign-flipping) and 20x scale-poisoning attacks under Non-IID client distributions while enforcing Local Differential Privacy (LDP).

---

## System Architecture

```text
       +-------------------------------------------------------------+
       |                  Subnet Traffic Multi-Graph                 |
       |             Nodes: IP Endpoints | Edges: NetFlows           |
       +-------------------------------------------------------------+
                                      │
                                      ▼
       +-------------------------------------------------------------+
       |               Spatial Feature Encoder (GATv2)               |
       |      Dynamic Multi-Head Topological Neighbor Attention      |
       +-------------------------------------------------------------+
                                      │
                                      ▼
       +-------------------------------------------------------------+
       |               Temporal State Recurrence (GRU)               |
       |        Sequential Hidden State Propagation Across Flows     |
       +-------------------------------------------------------------+
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
       +───────────────────────+               +───────────────────────+
       |  Engine 1: Classifier |               | Engine 2: Autoencoder |
       | Calibrated θ = 0.4050 |               | Unsupervised Anomaly  |
       | 91.14% Threat Capture |               |  Zero-Day Inspection  |
       +───────────────────────+               +───────────────────────+
                                      │
                                      ▼
       +-------------------------------------------------------------+
       |            Central Aggregator: ACS Defense Engine           |
       |  Coordinate-wise Median Consensus & Directional Alignment   |
       +-------------------------------------------------------------+


--- ## Empirical Benchmark Results (UNSW-NB15) 
Evaluated on the authentic **UNSW-NB15** benchmark across 24,000 independent testing flows (7,690 normal flows and 16,310 real attack incursions) partitioned under non-IID conditions.

### 1. Comparative Baseline Performance

| Paradigm | Method | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Supervised Centralized** | Centralized Random Forest (Tabular) | 89.28% | 0.9819 | 85.81% | 0.9158 | 0.9833 |
| **Decentralized Silo** | Local-Only GATv2-GRU (Isolated Subnet) | 70.62% | 0.9690 | 58.64% | 0.7306 | 0.7972 |
| **Federated Static** | Spatial Fed-GCN (Static Graph) | 69.35% | 0.9715 | 56.56% | 0.7150 | 0.8380 |
| **Proposed Framework** | **Fed-GNIDS (Dual-Engine, $\\theta = 0.4050$)** | **91.14%** | **0.8984** | **98.06%** | **0.9377** | **0.9620** |

*At an operational decision threshold of $\\theta = 0.4050$, Fed-GNIDS captures **15,993 out of 16,310 attacks** (98.06% recall) while preserving decentralized privacy.*

---

### 2. Byzantine Poisoning Defense ($K=8$ Subnets, $f=25\\%$ Adversaries)

| Defense Strategy & Attack Scenario | Accuracy | Recall | F1-Score | ROC-AUC | Defense Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline (No Attack)** | **91.14%** | **98.06%** | **0.9377** | **0.9620** | Baseline |
| **FedAvg under Sign-Flip Attack** | 82.93% | 85.61% | 0.8466 | 0.8886 | Degraded |
| **FedAvg under Scale-Poisoning ($20\\times$)** | **43.55%** | **59.87%** | **0.5904** | **0.3440** | **Collapsed** |
| **ACS under Sign-Flip (Proposed)** | **91.02%** | **97.85%** | **0.9366** | **0.9615** | **Resilient** |
| **ACS under Scale-Poison ($20\\times$) (Proposed)** | **90.91%** | **97.72%** | **0.9355** | **0.9609** | **Neutralized** |

---

### 3. Modular Ablation Analysis

| Architecture Configuration | ROC-AUC | Accuracy | Recall | F1-Score | Analytical Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Fed-GNIDS (GATv2 + GRU + DP)** | **0.9620** | **91.14%** | **98.06%** | **0.9377** | **Optimal spatial-temporal baseline** |
| **w/o Temporal Recurrence (Spatial Only)** | 0.9106 | 83.32% | 85.61% | 0.8395 | Recall drops by -12.45% without flow memory |
| **w/o Dynamic Attention (Static GCN + GRU)** | 0.9209 | 85.71% | 89.94% | 0.8723 | Static graphs miss dynamic edge relationships |
| **w/o Differential Privacy (No Norm Clip)** | 0.9500 | 89.29% | 95.96% | 0.9182 | Susceptible to gradient inversion leakage |

---

### 4. Edge Computational Footprint (AMD Ryzen 5 5500U Profile)

- **Model Parameters:** 68,848 trainable parameters (RAM size: **268.94 KB**)
- **Mean Per-Flow Latency:** **62.33 microseconds** per flow ($P95 = 17.33$ ms per 250-flow window)
- **Line Throughput:** **16,043 flows / second** on commodity host CPU
- **FL Network Transmission:** 268.94 KB uplink/downlink per round (4.20 MB total across 8 communication rounds)

---

 ## Repository Directory Structure



```text
Batch15_GNIDS/
├── data/
│   ├── raw/                           # Raw UNSW-NB15 flow CSV datasets
│   └── processed/                     # Dirichlet Non-IID graph partitions (.pt)
├── dashboard/
│   └── app.py                         # Streamlit Dual-Engine SOC Operations Center
├── results/
│   ├── ieee_baseline_comparison.png   # 300 DPI baseline benchmark plot
│   ├── ieee_confusion_matrix.png      # 300 DPI 98.06% recall confusion matrix
│   ├── ieee_roc_curve.png             # 300 DPI ROC curve (AUC = 0.9620)
│   ├── benchmark_comparison.csv       # Baseline comparison results
│   ├── adversarial_robustness_benchmark.csv # Byzantine defense metrics
│   └── ieee_ablation_results.csv      # Component ablation results
├── src/
│   ├── hybrid_pipeline.py             # Dual-Engine GATv2-GRU architecture
│   ├── server.py                      # Flower FL server with ACS aggregation
│   ├── client.py                      # Flower client with DP clipping & attack modes
│   ├── baselines.py                   # Comparative baseline evaluation runner
│   ├── simulate_attacks.py            # Byzantine robustness simulator
│   ├── ablation.py                    # Modular ablation analysis script
│   ├── calibrate_eval.py              # Decision boundary calibration engine
│   ├── profiler.py                    # Edge latency & memory profiler
│   └── verify_realtime.py             # Real-time streaming ingress simulator
├── paper.tex                          # Camera-ready IEEE double-column LaTeX manuscript
├── run_pipeline.py                    # Master end-to-end automated orchestrator
└── requirements.txt                   # Production environment dependencies
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

### 3. Real-Time Streaming Ingress Profiler
Evaluates line-rate sequential flow processing latency across 11 temporal telemetry windows:
```powershell
python src/verify_realtime.py
```

### 4. Launch the Streamlit SOC Operations Radar
Opens the dual-engine edge monitoring dashboard with interactive threshold calibration and live streaming simulations:
```powershell
streamlit run dashboard/app.py
```

---

## Authors & Academic Affiliation
* **Dilli Dinesh Edam** (Reg: 234M1A0530)
* **D Mahesh Babu** (Reg: 234M1A0528)
* **G Praveen** (Reg: 234M1A0535)
* **B Premchand** (Reg: 234M1A0520)
* **Dasari Kushvanth Kumar** (Reg: 234M1A0529)

**Project Guide:** Mr. K. Niranjan, Assistant Professor  
*Department of Computer Science and Engineering*  
**VEMU Institute of Technology**, Chittoor, Andhra Pradesh, India

---

## Citation
If you find this work useful in your research, cite:
```bibtex
@inproceedings{edam2026fedgnids,
  title={Fed-GNIDS: Byzantine-Robust Federated Spatial-Temporal Graph Neural Networks for Line-Rate Intrusion Detection at the Edge},
  author={Edam, Dilli Dinesh and Kumar, D Mahesh and Praveen, G and Premchand, B and Kumar, Dasari Kushvanth and Niranjan, K},
  booktitle={Proceedings of the IEEE Conference on Communications and Network Security},
  year={2026}
}
```