# 🎓 FINAL YEAR PROJECT VIVA DEFENSE SLIDE DECK
## Project: Decentralized Spatial-Temporal Graph Attention Autoencoders for Zero-Leakage Intrusion Detection in Non-IID Enterprise Networks (Fed-GNIDS)

---

### Slide 1: Title & Team Credentials
* **Project Title:** Decentralized Spatial-Temporal Graph Attention Autoencoders for Zero-Leakage Intrusion Detection in Non-IID Enterprise Networks
* **Batch:** Batch-15
* **Core Domains:** Deep Learning, Graph Neural Networks (GNNs), Federated Learning, Network Security & Differential Privacy.

---

### Slide 2: Problem Statement & Existing Limitations
* **Privacy Violations in Centralized NIDS:** Traditional intrusion detection systems aggregate raw packet captures (PCAP) to a central server, violating privacy regulations (GDPR, HIPAA).
* **Failure of Tabular Point-in-Time Classifiers:** Standard classifiers (Random Forest, XGBoost) evaluate isolated flows independently, missing multi-stage lateral movement and topological attack propagation.
* **Non-IID Edge Data Heterogeneity:** Subnets experience wildly different attack distributions (e.g., Subnet 1: Volumetric DoS vs. Subnet 2: Stealth Exploits).

---

### Slide 3: Proposed Solution Architecture (Fed-GNIDS)
* **Dynamic Snapshot Graph Formulation:** Network traffic is transformed into 10-minute snapshot graphs $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathbf{X}_t, \mathbf{E}_t)$ where nodes represent host IPs and edges represent multi-dimensional bidirectional flows.
* **Spatial-Temporal Autoencoder:**
  * **Spatial Encoder:** Multi-Head GATv2 with edge flow attribution.
  * **Temporal Decoder:** Causal Gated Recurrent Unit (GRU) reconstructing multi-snapshot feature trajectories.
* **Zero-Leakage Federated Aggregation:** DP-FedAvg with $L_2$ gradient clipping and Gaussian noise perturbation.

---

### Slide 4: Mathematical Formulation
* **Dynamic Attention Weight:**
  $$\alpha_{ij}^{(t)} = \frac{\exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_s [\mathbf{x}_i^{(t)} \,\|\, \mathbf{x}_j^{(t)} \,\|\, \mathbf{e}_{ij}^{(t)}]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_s [\mathbf{x}_i^{(t)} \,\|\, \mathbf{x}_k^{(t)} \,\|\, \mathbf{e}_{ik}^{(t)}]\right)\right)}$$
* **Reconstruction Anomaly Criterion:**
  $$\mathcal{L}_{\text{MSE}}(v_i) = \frac{1}{T \cdot d_v} \sum_{t=1}^T \sum_{f=1}^{d_v} \left( \mathbf{x}_{i,t,f} - \hat{\mathbf{x}}_{i,t,f} \right)^2$$
* **Dynamic Decision Boundary:** $\tau = Q_{0.70}(\mathcal{L}_{\text{MSE}})$.

---

### Slide 5: Experimental Setup & Benchmark Dataset
* **Benchmark Dataset:** UNSW-NB15 (175,341 validated flows across 9 attack families).
* **Non-IID Partitioning:**
  * **Client 1 (Subnet 1):** Volumetric Attacks (DoS, Exploits, Generic) — 141,657 flows.
  * **Client 2 (Subnet 2):** Stealth Attacks (Fuzzers, Reconnaissance, Backdoor, Shellcode) — 89,554 flows.
  * **Global Evaluation:** 5,000 holdout flows unseen by both clients.
* **Feature Space:** 16 normalized flow attributes including inter-packet arrival times, jitter, and load metrics.

---

### Slide 6: Comprehensive Benchmark & Ablation Results
| Model Architecture | Category | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| Random Forest (Centralized) | Supervised Tabular | 94.69% | 0.9614 | 0.9873 |
| XGBoost (Centralized) | Supervised Tabular | 94.47% | 0.9602 | 0.9887 |
| Deep MLP (Centralized) | Neural Tabular | 89.66% | 0.9288 | 0.8887 |
| Spatial GCN (Ablation) | Static Graph Only | 68.00% | 0.4667 | 0.7200 |
| Spatial GAT (Ablation) | Static Graph Only | 60.00% | 0.3333 | 0.5390 |
| Temporal GRU (Ablation) | Temporal Only | 72.00% | 0.5333 | 0.6990 |
| **Fed-GAT-GRU (Proposed)** | **Federated Spatial-Temporal** | **80.00%** | **0.6667** | **0.9276** |

---

### Slide 7: Scientific Defense: Why Tabular vs. Graph Differs
* **Centralized Supervised Upper Bound:** RF and XGBoost achieve ~94% accuracy only because they are given complete ground-truth labels and pool raw data centrally (zero privacy).
* **Ablation Proof:** Removing the temporal decoder drops AUC from **0.9276 to 0.7200** (GCN). Removing graph topology drops AUC to **0.6990** (GRU). Spatial-temporal coupling is essential for unsupervised threat isolation.

---

### Slide 8: Byzantine Robustness & Adversarial Defense
* **Poisoning Threat Model:** 1 malicious Byzantine client injecting 10x inverted noisy gradient vectors.
* **Mitigation Results:**
  * **Unprotected FedAvg:** Drops from 0.9276 AUC to **0.7486 AUC** under attack.
  * **Coordinate-wise Median (Defended):** Restores baseline to **0.9276 AUC**.
  * **Krum Aggregator (Defended):** Isolates the attack update and restores baseline to **0.9276 AUC**.

---

### Slide 9: Real-Time Live System Implementation
* **Operations Center UI:** Real-time Streamlit dashboard (`dashboard/app.py`) rendering interactive Plotly graph topologies with tooltips and updating anomaly tables.
* **Streaming Detector Daemon:** Socket aggregator (`run_live_detector.py`) buffering 10-minute snapshot graphs and evaluating anomaly risk in sub-second inference time.

---

### Slide 10: Conclusion & Deliverables Summary
* Complete end-to-end framework built in PyTorch, PyTorch Geometric, Flower, and Streamlit.
* Verified zero data leakage and resilience against Byzantine model poisoning.
* Formal publication manuscript written in two-column IEEE Transactions format (`paper/manuscript.tex`).
