import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from model import SpatialTemporalGNNAutoencoder

def train_local_epoch(model, dataset, optimizer, criterion, device, dp_clip=1.0):
    model.train()
    total_loss = 0.0
    count = 0
    for seq in dataset:
        seq = [snap.to(device) for snap in seq]
        target_snap = seq[-1]
        benign_mask = (target_snap.y == 0)
        if benign_mask.sum() == 0:
            continue
        optimizer.zero_grad()
        pred, target = model(seq)
        loss = criterion(pred[benign_mask], target[benign_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=dp_clip)
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(1, count)

def evaluate_model(model, test_dataset, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for seq in test_dataset:
            seq = [snap.to(device) for snap in seq]
            target_snap = seq[-1]
            rec, gt = model(seq)
            mse = torch.mean((rec - gt) ** 2, dim=-1)
            all_scores.extend(mse.cpu().numpy())
            all_labels.extend(target_snap.y.cpu().numpy())
            
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels, all_scores)
    normal_scores = all_scores[all_labels == 0]
    tau = np.mean(normal_scores) + 2.0 * np.std(normal_scores)
    preds = (all_scores >= tau).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    return roc_auc, acc, prec, rec, f1

def aggregate_fedavg(global_weights, client_deltas):
    avg_delta = [np.mean([cd[i] for cd in client_deltas], axis=0) for i in range(len(global_weights))]
    return [gw + ad for gw, ad in zip(global_weights, avg_delta)]

def aggregate_acs(global_weights, client_deltas, norm_threshold=5.0):
    flat_deltas = [np.concatenate([d.flatten() for d in cd]) for cd in client_deltas]
    median_ref = np.median(flat_deltas, axis=0)
    ref_norm = np.linalg.norm(median_ref) + 1e-8

    weights = []
    for f_delta in flat_deltas:
        l2_norm = np.linalg.norm(f_delta)
        norm_factor = min(1.0, norm_threshold / (l2_norm + 1e-8))
        cosine_sim = np.dot(f_delta, median_ref) / ((l2_norm * ref_norm) + 1e-8)
        alignment_factor = max(0.0, float(cosine_sim))
        weights.append(alignment_factor * norm_factor)

    sum_w = sum(weights) + 1e-8
    norm_weights = [w / sum_w for w in weights]

    aggregated_delta = [np.zeros_like(gw) for gw in global_weights]
    for client_idx, w in enumerate(norm_weights):
        for layer_idx, layer_delta in enumerate(client_deltas[client_idx]):
            aggregated_delta[layer_idx] += w * layer_delta

    return [gw + ad for gw, ad in zip(global_weights, aggregated_delta)]

def run_adversarial_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Byzantine Poisoning Benchmark (K=4 clients, f=25% Adversary) on {device}...")

    # Load partitions and create 4 subnets
    c0 = torch.load("data/processed/client_0_data.pt", weights_only=False)
    c1 = torch.load("data/processed/client_1_data.pt", weights_only=False)
    test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)

    # Split into 4 distinct non-IID client subnets
    half_c1 = len(c1) // 3
    client_pools = [
        c0,                      # Client 0 (Honest)
        c1[:half_c1],            # Client 1 (Honest)
        c1[half_c1: 2*half_c1],  # Client 2 (Honest)
        c1[2*half_c1:]           # Client 3 (Adversary: Byzantine Subnet)
    ]

    scenarios = [
        ("Clean Baseline (No Attack)", "fedavg", "none"),
        ("FedAvg under Sign-Flip Attack", "fedavg", "sign_flip"),
        ("FedAvg under Scale-Poisoning (20x)", "fedavg", "scale_poison"),
        ("ACS under Sign-Flip Attack (Proposed)", "acs", "sign_flip"),
        ("ACS under Scale-Poisoning (Proposed)", "acs", "scale_poison"),
    ]

    results = []

    for name, strategy, attack_type in scenarios:
        print(f"\n[+] Testing Scenario: {name}...")
        torch.manual_seed(42)
        global_model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(device)
        global_weights = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        criterion = nn.MSELoss()

        for r in range(1, 4):
            client_deltas = []
            for c_id, pool in enumerate(client_pools):
                c_model = copy.deepcopy(global_model)
                opt = torch.optim.Adam(c_model.parameters(), lr=0.005)
                train_local_epoch(c_model, pool, opt, criterion, device)
                c_weights = [val.cpu().numpy() for _, val in c_model.state_dict().items()]
                delta = [cw - gw for cw, gw in zip(c_weights, global_weights)]

                # Client 3 is the compromised Byzantine node
                if c_id == 3:
                    if attack_type == "sign_flip":
                        delta = [-1.0 * d for d in delta]
                    elif attack_type == "scale_poison":
                        delta = [20.0 * d for d in delta]

                client_deltas.append(delta)

            if strategy == "fedavg":
                global_weights = aggregate_fedavg(global_weights, client_deltas)
            else:
                global_weights = aggregate_acs(global_weights, client_deltas, norm_threshold=5.0)

            params_dict = zip(global_model.state_dict().keys(), [torch.tensor(w) for w in global_weights])
            global_model.load_state_dict({k: v for k, v in params_dict})

        auc, acc, prec, rec, f1 = evaluate_model(global_model, test_data, device)
        print(f"    --> Results: ROC-AUC={auc:.4f} | Accuracy={acc*100:.2f}% | Precision={prec:.4f} | Recall={rec*100:.2f}% | F1={f1:.4f}")
        results.append({
            "Defense Strategy & Attack Scenario": name,
            "Strategy": strategy.upper(),
            "Attack Type": attack_type,
            "ROC-AUC": auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    df_results = pd.DataFrame(results)
    print("\n" + "="*90)
    print("      IEEE ADVERSARIAL POISONING ROBUSTNESS BENCHMARK TABLE (K=4, f=25%)")
    print("="*90)
    print(df_results[["Defense Strategy & Attack Scenario", "ROC-AUC", "Accuracy", "Recall", "F1-Score"]].to_string(index=False))
    print("="*90)

    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/adversarial_robustness_benchmark.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")
    sns.barplot(
        data=df_results,
        x="Defense Strategy & Attack Scenario",
        y="ROC-AUC",
        hue="Defense Strategy & Attack Scenario",
        legend=False,
        palette=["#2ca02c", "#d62728", "#d62728", "#1f77b4", "#1f77b4"]
    )
    plt.xticks(rotation=20, ha='right', fontsize=9, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.title("Byzantine Poisoning Robustness: FedAvg Collapse vs. ACS Resilience (f=25%)", fontsize=12, fontweight='bold')
    plt.axhline(0.5, color='gray', linestyle='--', label='Random Guess Baseline')
    plt.tight_layout()
    plt.savefig("results/ieee_adversarial_robustness.png", dpi=300)
    plt.close()
    print("[+] Updated IEEE Publication Figure: results/ieee_adversarial_robustness.png")

if __name__ == "__main__":
    run_adversarial_simulation()