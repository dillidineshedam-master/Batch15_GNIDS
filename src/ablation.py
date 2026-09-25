import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import GATv2Conv, GCNConv

# Variant 1: Spatial Only (No GRU temporal modeling)
class SpatialOnlyGATv2Autoencoder(nn.Module):
    def __init__(self, in_node_dim=50, edge_dim=16, hidden_dim=64):
        super(SpatialOnlyGATv2Autoencoder, self).__init__()
        self.gat1 = GATv2Conv(in_node_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )

    def forward(self, seq):
        target_snap = seq[-1]
        h = torch.relu(self.gat1(target_snap.x, target_snap.edge_index, target_snap.edge_attr))
        h = self.gat2(h, target_snap.edge_index, target_snap.edge_attr)
        src, dst = target_snap.edge_index[0], target_snap.edge_index[1]
        pair = torch.cat([h[src], h[dst]], dim=-1)
        return self.edge_decoder(pair), target_snap.edge_attr

# Variant 2: GCN + GRU (Replaces GATv2 with isotropic standard GCN)
class GCNGRUAutoencoder(nn.Module):
    def __init__(self, in_node_dim=50, edge_dim=16, hidden_dim=64):
        super(GCNGRUAutoencoder, self).__init__()
        self.gcn = GCNConv(in_node_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )

    def forward(self, seq):
        h_seq = []
        for snap in seq:
            h_snap = torch.relu(self.gcn(snap.x, snap.edge_index))
            h_seq.append(h_snap)
        stacked = torch.stack(h_seq, dim=1)
        gru_out, _ = self.gru(stacked)
        h_final = gru_out[:, -1, :]
        target_snap = seq[-1]
        src, dst = target_snap.edge_index[0], target_snap.edge_index[1]
        pair = torch.cat([h_final[src], h_final[dst]], dim=-1)
        return self.edge_decoder(pair), target_snap.edge_attr

def train_and_eval(model, train_data, test_data, device, dp_clip=1.0, epochs=3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for seq in train_data:
            seq = [s.to(device) for s in seq]
            target_snap = seq[-1]
            mask = (target_snap.y == 0)
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            rec, gt = model(seq)
            loss = criterion(rec[mask], gt[mask])
            loss.backward()
            if dp_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=dp_clip)
            optimizer.step()

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for seq in test_data:
            seq = [s.to(device) for s in seq]
            rec, gt = model(seq)
            mse = torch.mean((rec - gt) ** 2, dim=-1)
            scores.extend(mse.cpu().numpy())
            labels.extend(seq[-1].y.cpu().numpy())

    scores = np.array(scores)
    labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    tau = np.mean(scores[labels == 0]) + 2.0 * np.std(scores[labels == 0])
    preds = (scores >= tau).astype(int)

    return {
        "ROC-AUC": auc,
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1-Score": f1_score(labels, preds, zero_division=0)
    }

def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running IEEE Ablation Study on {device}...")

    train_data = torch.load("data/processed/client_0_data.pt", weights_only=False)
    test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)

    from model import SpatialTemporalGNNAutoencoder

    ablations = [
        ("Full Architecture (GATv2 + GRU + DP)", SpatialTemporalGNNAutoencoder(50, 16, 64), 1.0),
        ("w/o Temporal Recurrence (Spatial GATv2 Only)", SpatialOnlyGATv2Autoencoder(50, 16, 64), 1.0),
        ("w/o Graph Attention (Static GCN + GRU)", GCNGRUAutoencoder(50, 16, 64), 1.0),
        ("w/o Differential Privacy (No Norm Clipping)", SpatialTemporalGNNAutoencoder(50, 16, 64), None),
    ]

    results = []
    for name, model, clip in ablations:
        print(f"\n[+] Evaluating Variant: {name}...")
        torch.manual_seed(42)
        metrics = train_and_eval(model, train_data, test_data, device, dp_clip=clip)
        print(f"    --> ROC-AUC: {metrics['ROC-AUC']:.4f} | F1: {metrics['F1-Score']:.4f} | Recall: {metrics['Recall']*100:.2f}%")
        results.append({"Configuration": name, **metrics})

    df = pd.DataFrame(results)
    print("\n" + "="*85)
    print("                    IEEE ABLATION ANALYSIS TABLE")
    print("="*85)
    print(df[["Configuration", "ROC-AUC", "Accuracy", "Recall", "F1-Score"]].to_string(index=False))
    print("="*85)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ieee_ablation_results.csv", index=False)
    print("[+] Saved results to: results/ieee_ablation_results.csv")

    plt.figure(figsize=(9, 4.5))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x="Configuration", y="F1-Score", hue="Configuration", legend=False, palette="Blues_r")
    plt.xticks(rotation=15, ha='right', fontsize=9, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1-Score", fontsize=11, fontweight='bold')
    plt.title("Ablation Study: Impact of Topological and Temporal Sub-Modules", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/ieee_ablation_study.png", dpi=300)
    plt.close()
    print("[+] Saved Figure: results/ieee_ablation_study.png")

if __name__ == "__main__":
    run_ablation()