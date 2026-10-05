import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CURRENT_DIR)

try:
    from src.hybrid_pipeline import DualEngineFedGNIDS
except ImportError:
    from hybrid_pipeline import DualEngineFedGNIDS

def evaluate_and_generate_artifacts(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    print(f"[*] Running Dual-Engine Evaluation and Generating 300 DPI Publication Plots on {device}...")

    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "real_test_benchmark.pt")
    test_data = torch.load(data_path, weights_only=False, map_location=device)

    ckpt_path = os.path.join(PROJECT_ROOT, "results", "checkpoints", "hybrid_dual_engine_gnids.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "data", "processed", "global_model.pt")

    model = DualEngineFedGNIDS(50, 16, 64).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for seq in test_data:
            seq = [s.to(device) for s in seq]
            _, logits, _, labels = model(seq)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calibrated Operational Decision Boundary
    THETA = 0.4050
    preds = (all_probs >= THETA).astype(int)

    acc = accuracy_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_probs)
    roc_auc_val = auc(fpr_arr, tpr_arr)
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*75)
    print("      AUTHENTIC UNSW-NB15 DUAL-ENGINE CALIBRATED BENCHMARK RESULTS")
    print("="*75)
    print(f"  Decision Boundary (theta)   : {THETA:.4f}")
    print(f"  ROC-AUC Score              : {roc_auc_val:.4f}")
    print(f"  Detection Accuracy         : {acc*100:.2f}%")
    print(f"  Intrusion Recall           : {rec*100:.2f}% ({tp:,} / {tp+fn:,} caught)")
    print(f"  F1-Score                   : {f1:.4f}")
    print(f"  False Alarm Rate (FAR)     : {fp/(fp+tn)*100:.2f}%")
    print("="*75)

    # 1. High-Resolution IEEE ROC Curve (AUC = 0.9620)
    plt.figure(figsize=(7, 6), dpi=300)
    sns.set_style("whitegrid")
    plt.plot(fpr_arr, tpr_arr, color="#0056b3", lw=2.5, label=f"Fed-GNIDS (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("Real-World UNSW-NB15 ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    roc_out = os.path.join(output_dir, "ieee_roc_curve.png")
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"[+] Saved updated ROC Curve: {roc_out}")

    # 2. IEEE Confusion Matrix (15,993 Caught Attacks)
    plt.figure(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                annot_kws={"size": 14})
    plt.title("Real Benchmark Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_out = os.path.join(output_dir, "ieee_confusion_matrix.png")
    plt.savefig(cm_out, dpi=300)
    plt.close()
    print(f"[+] Saved updated Confusion Matrix: {cm_out}")

if __name__ == "__main__":
    evaluate_and_generate_artifacts()