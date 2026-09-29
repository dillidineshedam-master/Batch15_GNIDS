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

def train_client_local(model, data_slice, device, epochs=3, lr=0.005, dp_clip=1.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for _ in range(epochs):
        for seq in data_slice:
            seq = [s.to(device) for s in seq]
            mask = (seq[-1].y == 0)
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            rec, gt = model(seq)
            loss = criterion(rec[mask], gt[mask])
            loss.backward()
            if dp_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=dp_clip)
            optimizer.step()
            
    return model.state_dict()

def evaluate_model(model, test_data, device):
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

def aggregate_fedavg(global_state, client_states):
    new_state = copy.deepcopy(global_state)
    for key in new_state.keys():
        stacked = torch.stack([cs[key].float() for cs in client_states], dim=0)
        new_state[key] = torch.mean(stacked, dim=0).to(new_state[key].dtype)
    return new_state

def aggregate_acs(global_state, client_states, clip_norm=5.0):
    new_state = copy.deepcopy(global_state)
    deltas = []
    flattened_deltas = []

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

def run_byzantine_benchmark(num_clients=8, num_byzantine=2, rounds=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Scaled Byzantine Benchmark (K={num_clients} clients, f={num_byzantine/num_clients*100:.0f}% Adversaries) on {device}...")

    c0 = torch.load("data/processed/client_0_data.pt", weights_only=False)
    c1 = torch.load("data/processed/client_1_data.pt", weights_only=False)
    pool = c0 + c1
    test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)

    # Partition across K=8 subnets
    chunk_size = len(pool) // num_clients
    subnets = [pool[i * chunk_size:(i + 1) * chunk_size] for i in range(num_clients)]

    scenarios = [
        ("Clean Baseline (No Attack)", "fedavg", "none"),
        ("FedAvg under Sign-Flip Attack", "fedavg", "sign_flip"),
        ("FedAvg under Scale-Poisoning (20x)", "fedavg", "scale_20x"),
        ("ACS under Sign-Flip Attack (Proposed)", "acs", "sign_flip"),
        ("ACS under Scale-Poisoning (Proposed)", "acs", "scale_20x"),
    ]

    results = []

    for name, agg_mode, attack_mode in scenarios:
        print(f"\n[+] Testing Scenario: {name}...")
        torch.manual_seed(42)
        global_model = SpatialTemporalGNNAutoencoder(50, 16, 64).to(device)

        for rnd in range(rounds):
            client_states = []
            for k in range(num_clients):
                local_m = copy.deepcopy(global_model)
                local_state = train_client_local(local_m, subnets[k], device, epochs=2)

                # First num_byzantine clients are Byzantine adversaries
                if k < num_byzantine and attack_mode != "none":
                    corrupted = copy.deepcopy(local_state)
                    for key in corrupted.keys():
                        delta = local_state[key].float() - global_model.state_dict()[key].float()
                        if attack_mode == "sign_flip":
                            corrupted[key] = (global_model.state_dict()[key].float() - delta).to(corrupted[key].dtype)
                        elif attack_mode == "scale_20x":
                            corrupted[key] = (global_model.state_dict()[key].float() + 20.0 * delta).to(corrupted[key].dtype)
                    client_states.append(corrupted)
                else:
                    client_states.append(local_state)

            if agg_mode == "fedavg":
                new_state = aggregate_fedavg(global_model.state_dict(), client_states)
            else:
                new_state = aggregate_acs(global_model.state_dict(), client_states)

            global_model.load_state_dict(new_state)

        metrics = evaluate_model(global_model, test_data, device)
        print(f"    --> Results: ROC-AUC={metrics['ROC-AUC']:.4f} | Accuracy={metrics['Accuracy']*100:.2f}% | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']*100:.2f}% | F1={metrics['F1-Score']:.4f}")
        results.append({"Defense Strategy & Attack Scenario": name, **metrics})

    df = pd.DataFrame(results)
    print("\n" + "="*90)
    print(f"      IEEE ADVERSARIAL POISONING ROBUSTNESS BENCHMARK TABLE (K={num_clients}, f={num_byzantine/num_clients*100:.0f}%)")
    print("="*90)
    print(df[["Defense Strategy & Attack Scenario", "ROC-AUC", "Accuracy", "Recall", "F1-Score"]].to_string(index=False))
    print("="*90)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/adversarial_robustness_benchmark.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x="Defense Strategy & Attack Scenario", y="Accuracy", hue="Defense Strategy & Attack Scenario", legend=False, palette="Set1")
    plt.xticks(rotation=20, ha='right', fontsize=9, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.ylabel("Detection Accuracy", fontsize=11, fontweight='bold')
    plt.title(f"Byzantine Defense Scalability: FedAvg vs ACS (K={num_clients} Clients, f=25% Adversaries)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/ieee_adversarial_robustness.png", dpi=300)
    plt.close()
    print("[+] Updated IEEE Publication Figure: results/ieee_adversarial_robustness.png")

if __name__ == "__main__":
    run_byzantine_benchmark(num_clients=8, num_byzantine=2, rounds=4)