import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
from model import SpatialTemporalGNNAutoencoder

def evaluate_global_model(model_path="data/processed/global_model.pt", test_data_path="data/processed/real_test_benchmark.pt", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[-] Global model not found at {model_path}.")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"[-] Benchmark test dataset not found at {test_data_path}. Run preprocess.py first.")

    model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_dataset = torch.load(test_data_path, weights_only=False)
    print(f"[+] Evaluating on {len(test_dataset)} official benchmark temporal sequences...")

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

    num_normal = (all_labels == 0).sum()
    num_attacks = (all_labels == 1).sum()
    print(f"[+] Ground-Truth Evaluation Entities: {num_normal:,} Benign Flows, {num_attacks:,} Attack Flows")

    roc_auc = roc_auc_score(all_labels, all_scores)

    # Dynamic Threshold: mu + 2.0 * sigma on normal baseline
    normal_scores = all_scores[all_labels == 0]
    threshold = np.mean(normal_scores) + 2.0 * np.std(normal_scores)
    predictions = (all_scores >= threshold).astype(int)

    acc = accuracy_score(all_labels, predictions)
    prec = precision_score(all_labels, predictions, zero_division=0)
    rec = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    far = fp / (fp + tn + 1e-8)

    print("\n=======================================================")
    print("   AUTHENTIC UNSW-NB15 BENCHMARK EVALUATION RESULTS   ")
    print("=======================================================")
    print(f"  Anomaly Threshold (tau) : {threshold:.6f}")
    print(f"  ROC-AUC Score          : {roc_auc:.4f}")
    print(f"  Detection Accuracy     : {acc * 100:.2f}%")
    print(f"  Precision              : {prec:.4f}")
    print(f"  Recall (Detection Rate): {rec * 100:.2f}%")
    print(f"  F1-Score               : {f1:.4f}")
    print(f"  False Alarm Rate (FAR) : {far * 100:.2f}%")
    print("=======================================================\n")

    # Save Publication Figures
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#0066cc', lw=2.5, label=f'Fed-GNIDS (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Real-World UNSW-NB15 ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ieee_roc_curve.png"), dpi=300)
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(5, 4.5))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Ground Truth Label', fontweight='bold')
    plt.title('Real Benchmark Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ieee_confusion_matrix.png"), dpi=300)
    plt.close()

    print("[+] Publication figures updated in results/")

if __name__ == "__main__":
    evaluate_global_model()