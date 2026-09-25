# Fed-GNIDS: Spatial-Temporal Graph Attention Autoencoders for Zero-Leakage Intrusion Detection

## 1. Project Overview & Architectural Novelty
Centralized Network Intrusion Detection Systems (NIDS) aggregate raw PCAP telemetry into a single location, violating privacy regulations and leaving networks vulnerable to multi-stage lateral attacks.

**Fed-GNIDS** resolves these challenges by introducing:
* **Dynamic Graph Snapshot Slicing:** Continuous conversion of network flows into 10-minute snapshot graphs $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathbf{X}_t, \mathbf{E}_t)$ capturing topological relations.
* **Spatial-Temporal Autoencoder:** Coupling edge-conditioned Multi-Head GATv2 with recurrent GRU sequence decoders.
* **Non-IID Differential Privacy (DP-FedAvg):** Local edge training with $L_2$ gradient clipping and Gaussian noise perturbation.
* **Byzantine Robustness:** Integrated Coordinate-wise Median and Krum aggregation strategies to mitigate model poisoning.

---

## 2. Experimental Benchmark & Ablation Matrix

| Model Architecture | Category | Accuracy (%) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Centralized Random Forest** | Tabular ML (Supervised) | 94.69% | 0.9614 | 0.9873 |
| **Centralized XGBoost** | Tabular ML (Supervised) | 94.47% | 0.9602 | 0.9887 |
| **Centralized Deep MLP** | Neural Tabular (Supervised) | 89.66% | 0.9288 | 0.8887 |
| **Spatial GCN** | Static Graph Ablation | 68.00% | 0.4667 | 0.7200 |
| **Spatial GAT** | Static Graph Ablation | 60.00% | 0.3333 | 0.5390 |
| **Temporal GRU** | Temporal Only Ablation | 72.00% | 0.5333 | 0.6990 |
| **Fed-GAT-GRU (Proposed)** | **Federated Spatial-Temporal** | **80.00%** | **0.6667** | **0.9276** |

---

## 3. Byzantine Attack & Defense Evaluation

| Aggregation Strategy | Accuracy (%) | F1-Score | ROC-AUC | Status Under Attack |
| :--- | :---: | :---: | :---: | :--- |
| **Unprotected FedAvg** | 76.00% | 0.6000 | 0.7486 | Degraded (-17.9% AUC drop) |
| **Coordinate-wise Median** | **80.00%** | **0.6667** | **0.9276** | **Fully Resilient** |
| **Trimmed Mean ($\beta=0.2$)** | 52.00% | 0.2000 | 0.3714 | Degraded ($N < 2k+1$) |
| **Krum Aggregator** | **80.00%** | **0.6667** | **0.9276** | **Fully Resilient** |

---

## 4. Execution Commands Reference

* **Live Streamlit Dashboard:** `python run_dashboard.py`
* **Live Network Card Detector:** `python run_live_detector.py`
* **Benchmark & Ablations Matrix:** `python -m evaluation.benchmark_suite`
* **Adversarial Byzantine Benchmark:** `python -m evaluation.adversarial_eval`
* **Generate Publication Figures:** `python -m evaluation.generate_plots`
