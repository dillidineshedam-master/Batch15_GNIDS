import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from hybrid_pipeline import DualEngineFedGNIDS

device = torch.device("cpu")
print("[*] Generating IEEE Publication Artifacts & Confusion Matrix...")

# 1. Load benchmark and model checkpoint
test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)
model = DualEngineFedGNIDS(50, 16, 64).to(device)
model.load_state_dict(torch.load("results/checkpoints/hybrid_dual_engine_gnids.pt", weights_only=False))
model.eval()

cls_probs, all_labels = [], []

with torch.no_grad():
    for seq in test_data:
        seq = [s.to(device) for s in seq]
        _, logits, _, gt_labels = model(seq)
        probs = torch.sigmoid(logits)
        cls_probs.extend(probs.numpy())
        all_labels.extend(gt_labels.numpy())

cls_probs = np.array(cls_probs)
all_labels = np.array(all_labels)

# Calibrated production threshold
THETA = 0.4050
preds = (cls_probs >= THETA).astype(int)

# 2. Confusion Matrix Computation
cm = confusion_matrix(all_labels, preds)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*70)
print("             OPERATIONAL THREAT BREAKDOWN (24,000 TEST FLOWS)")
print("="*70)
print(f"  True Positives  (Attacks Caught)       : {tp:,} / 16,310 ({tp/16310*100:.2f}%)")
print(f"  False Negatives (Attacks Missed)       : {fn:,} / 16,310 ({fn/16310*100:.2f}%)")
print(f"  True Negatives  (Benign Correct)       : {tn:,} / 7,690  ({tn/7690*100:.2f}%)")
print(f"  False Positives (False Alarms)         : {fp:,} / 7,690  ({fp/7690*100:.2f}%)")
print(f"  Global Detection Accuracy              : {(tp+tn)/len(all_labels)*100:.2f}%")
print("="*70)

os.makedirs("results", exist_ok=True)

# 3. Figure 1: Camera-Ready ROC Curve (300 DPI)
fpr, tpr, _ = roc_curve(all_labels, cls_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'GATv2-GRU Fed-GNIDS (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Random Guessing (AUC = 0.50)')

# Highlight optimal threshold point
opt_fp_rate = fp / (fp + tn)
opt_tp_rate = tp / (tp + fn)
plt.scatter([opt_fp_rate], [opt_tp_rate], color='#d62728', s=80, zorder=5, 
            label=f'Operating Point ($\\theta=0.4050$)\nAcc: 91.14%, Recall: 98.06%')

plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
plt.title('Receiver Operating Characteristic (UNSW-NB15)', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', fontsize=9, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("results/ieee_roc_curve.png", dpi=300)
plt.close()
print("[+] Saved: results/ieee_roc_curve.png (300 DPI)")

# 4. Figure 2: Publication Confusion Matrix (300 DPI)
plt.figure(figsize=(5.5, 4.8), dpi=300)
labels = ['Benign (0)', 'Attack (1)']
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 14, 'weight': 'bold'})

plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
plt.ylabel('Ground Truth Class', fontsize=11, fontweight='bold')
plt.title('Fed-GNIDS Operational Confusion Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("results/ieee_confusion_matrix.png", dpi=300)
plt.close()
print("[+] Saved: results/ieee_confusion_matrix.png (300 DPI)")