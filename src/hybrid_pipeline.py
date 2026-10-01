import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import GATv2Conv

# ==========================================
# 1. Dual-Engine Architecture Definition
# ==========================================
class DualEngineFedGNIDS(nn.Module):
    def __init__(self, in_node_dim=50, edge_dim=16, hidden_dim=64):
        super(DualEngineFedGNIDS, self).__init__()
        # Shared Spatial Attention Backbone
        self.gat1 = GATv2Conv(in_node_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
        
        # Shared Temporal Recurrence Backbone
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Engine 1: Unsupervised Reconstruction Decoder (Zero-Day MSE)
        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )
        
        # Engine 2: Supervised Intrusion Classification Head (Known Incursions BCE)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, seq):
        h_seq = []
        for snap in seq:
            h = torch.relu(self.gat1(snap.x, snap.edge_index, snap.edge_attr))
            h = self.gat2(h, snap.edge_index, snap.edge_attr)
            h_seq.append(h)
            
        stacked = torch.stack(h_seq, dim=1)
        gru_out, _ = self.gru(stacked)
        h_final = gru_out[:, -1, :]
        
        target_snap = seq[-1]
        src, dst = target_snap.edge_index[0], target_snap.edge_index[1]
        edge_repr = torch.cat([h_final[src], h_final[dst]], dim=-1)
        
        recon_edges = self.recon_decoder(edge_repr)
        logits = self.cls_head(edge_repr).squeeze(-1)
        
        return recon_edges, logits, target_snap.edge_attr, target_snap.y

# ==========================================
# 2. Local Multi-Task Training Engine
# ==========================================
def train_hybrid_client(model, train_data, device, epochs=3, lr=0.003, alpha=0.6, dp_clip=1.0):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
    for _ in range(epochs):
        for seq in train_data:
            seq = [s.to(device) for s in seq]
            rec_edges, logits, gt_edges, gt_labels = model(seq)
            
            # Unsupervised loss on benign baseline flows
            benign_mask = (gt_labels == 0)
            if benign_mask.sum() > 0:
                loss_recon = mse_criterion(rec_edges[benign_mask], gt_edges[benign_mask])
            else:
                loss_recon = torch.tensor(0.0, device=device)
                
            # Supervised loss across all flow labels
            loss_cls = bce_criterion(logits, gt_labels.float())
            
            # Joint Multi-Task Optimization
            total_loss = (alpha * loss_cls) + ((1.0 - alpha) * loss_recon)
            
            optimizer.zero_grad()
            total_loss.backward()
            if dp_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=dp_clip)
            optimizer.step()
            
    return model.state_dict()

# ==========================================
# 3. Byzantine-Robust ACS Aggregator
# ==========================================
def aggregate_acs(global_state, client_states, clip_norm=5.0):
    new_state = copy.deepcopy(global_state)
    deltas, flattened_deltas = [], []

    for cs in client_states:
        delta = {k: cs[k].float() - global_state[k].float() for k in cs.keys()}
        deltas.append(delta)
        flat = torch.cat([d.flatten() for d in delta.values()])
        flattened_deltas.append(flat)

    stacked_flats = torch.stack(flattened_deltas, dim=0)
    median_ref = torch.median(stacked_flats, dim=0).values
    median_norm = torch.norm(median_ref) + 1e-7

    weights = []
    for flat in flattened_deltas:
        u_norm = torch.norm(flat) + 1e-7
        beta = min(1.0, float(clip_norm / u_norm))
        cos_sim = float(torch.dot(flat, median_ref) / (u_norm * median_norm))
        gamma = max(0.0, cos_sim)
        weights.append(beta * gamma)

    total_w = sum(weights) + 1e-7
    norm_weights = [w / total_w for w in weights]

    for key in new_state.keys():
        weighted_delta = sum(norm_weights[i] * deltas[i][key] for i in range(len(client_states)))
        new_state[key] = (global_state[key].float() + weighted_delta).to(new_state[key].dtype)

    return new_state

# ==========================================
# 4. Joint Dual-Engine Evaluator
# ==========================================
def evaluate_dual_engine(model, test_data, device):
    model.eval()
    recon_errors, cls_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for seq in test_data:
            seq = [s.to(device) for s in seq]
            rec_edges, logits, gt_edges, gt_labels = model(seq)
            
            # Engine 1: Unsupervised reconstruction MSE
            mse = torch.mean((rec_edges - gt_edges) ** 2, dim=-1)
            recon_errors.extend(mse.cpu().numpy())
            
            # Engine 2: Supervised classification sigmoid probability
            probs = torch.sigmoid(logits)
            cls_probs.extend(probs.cpu().numpy())
            
            all_labels.extend(gt_labels.cpu().numpy())
            
    recon_errors = np.array(recon_errors)
    cls_probs = np.array(cls_probs)
    all_labels = np.array(all_labels)
    
    # Engine 1 Metrics (Unsupervised MSE @ tau = mu + 2*sigma)
    tau = np.mean(recon_errors[all_labels == 0]) + 2.0 * np.std(recon_errors[all_labels == 0])
    unsup_preds = (recon_errors >= tau).astype(int)
    unsup_metrics = {
        "ROC-AUC": roc_auc_score(all_labels, recon_errors),
        "Accuracy": accuracy_score(all_labels, unsup_preds),
        "Precision": precision_score(all_labels, unsup_preds, zero_division=0),
        "Recall": recall_score(all_labels, unsup_preds, zero_division=0),
        "F1-Score": f1_score(all_labels, unsup_preds, zero_division=0)
    }
    
    # Engine 2 Metrics (Supervised Classification @ threshold 0.5)
    sup_preds = (cls_probs >= 0.5).astype(int)
    sup_metrics = {
        "ROC-AUC": roc_auc_score(all_labels, cls_probs),
        "Accuracy": accuracy_score(all_labels, sup_preds),
        "Precision": precision_score(all_labels, sup_preds, zero_division=0),
        "Recall": recall_score(all_labels, sup_preds, zero_division=0),
        "F1-Score": f1_score(all_labels, sup_preds, zero_division=0)
    }
    
    return unsup_metrics, sup_metrics

# ==========================================
# 5. Master Pipeline Execution
# ==========================================
def run_hybrid_federation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Dual-Engine Federated Training (GATv2-GRU + ACS) on {device}...")

    c0 = torch.load("data/processed/client_0_data.pt", weights_only=False)
    c1 = torch.load("data/processed/client_1_data.pt", weights_only=False)
    test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)
    
    clients_data = [c0, c1]
    global_model = DualEngineFedGNIDS(50, 16, 64).to(device)
    
    rounds = 5
    for rnd in range(1, rounds + 1):
        print(f"\n[+] Executing Federated Multi-Task Round {rnd}/{rounds}...")
        client_states = []
        for cid, cdata in enumerate(clients_data):
            local_model = copy.deepcopy(global_model)
            state = train_hybrid_client(local_model, cdata, device, epochs=3, lr=0.003, alpha=0.6, dp_clip=1.0)
            client_states.append(state)
            
        new_state = aggregate_acs(global_model.state_dict(), client_states)
        global_model.load_state_dict(new_state)

    print("\n[*] Evaluating Both Engines on 24,000 Real UNSW-NB15 Test Flows...")
    unsup_m, sup_m = evaluate_dual_engine(global_model, test_data, device)
    
    results = [
        {"Operational Engine": "Engine 1: Supervised Classifier (Known Incursions)", **sup_m},
        {"Operational Engine": "Engine 2: Unsupervised Autoencoder (Zero-Day Baseline)", **unsup_m}
    ]
    
    df = pd.DataFrame(results)
    print("\n" + "="*95)
    print("                 DUAL-ENGINE FED-GNIDS EMPIRICAL BENCHMARK MATRIX")
    print("="*95)
    print(df[["Operational Engine", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))
    print("="*95)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    df.to_csv("results/ieee_hybrid_benchmarks.csv", index=False)
    
    # Save checkpoint
    torch.save(global_model.state_dict(), "results/checkpoints/hybrid_dual_engine_gnids.pt")
    print("[+] Model Checkpoint Saved: results/checkpoints/hybrid_dual_engine_gnids.pt")
    
    # Save plot
    plt.figure(figsize=(9, 4.5))
    sns.set_style("whitegrid")
    df_melt = pd.melt(df, id_vars=["Operational Engine"], value_vars=["Accuracy", "Recall", "F1-Score", "ROC-AUC"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Operational Engine", palette=["#2ecc71", "#3498db"])
    plt.ylim(0.0, 1.05)
    plt.ylabel("Performance Score", fontsize=11, fontweight='bold')
    plt.title("Dual-Engine Performance: Supervised Production Head vs. Zero-Knowledge Baseline", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/ieee_hybrid_performance.png", dpi=300)
    plt.close()
    print("[+] Publication Figure Saved: results/ieee_hybrid_performance.png")

if __name__ == "__main__":
    run_hybrid_federation()