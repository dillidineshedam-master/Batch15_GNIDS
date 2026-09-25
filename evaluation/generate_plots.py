import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/figures"
METRICS_CSV = "results/metrics/comprehensive_ieee_benchmark_matrix.csv"
FED_CSV = "results/metrics/federated_training_metrics.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_ablation_barchart():
    if not os.path.exists(METRICS_CSV):
        print(f"[-] Missing {METRICS_CSV}")
        return

    df = pd.read_csv(METRICS_CSV)
    
    # Filter graph models and proposed framework
    models = df["Model Architecture"].tolist()
    auc_scores = df["ROC-AUC"].tolist()
    f1_scores = df["F1-Score"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width/2, auc_scores, width, label='ROC-AUC', color='#2980b9', edgecolor='black')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#e67e22', edgecolor='black')

    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation & Architectural Comparison on UNSW-NB15', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 1.1)

    # Attach score values above bars
    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "ieee_ablation_barchart.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ablation chart: {save_path}")

def generate_convergence_curves():
    if not os.path.exists(FED_CSV):
        print(f"[-] Missing {FED_CSV}")
        return

    df = pd.read_csv(FED_CSV)
    rounds = df["round"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: Loss Curve
    ax1.plot(rounds, df["loss"], marker='o', linewidth=2, color='#c0392b', label='Reconstruction MSE')
    ax1.set_xlabel('Federated Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Global Loss (MSE)', fontsize=11, fontweight='bold')
    ax1.set_title('Global Autoencoder Convergence (DP-FedAvg)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot 2: Metrics Curve
    ax2.plot(rounds, df["roc_auc"], marker='s', linewidth=2, color='#27ae60', label='ROC-AUC')
    ax2.plot(rounds, df["accuracy"], marker='^', linewidth=2, color='#2980b9', label='Accuracy')
    ax2.plot(rounds, df["f1_score"], marker='d', linewidth=2, color='#8e44ad', label='F1-Score')
    ax2.set_xlabel('Federated Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score Metric', fontsize=11, fontweight='bold')
    ax2.set_title('Federated Evaluation Metrics Over Rounds', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "ieee_federated_convergence.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved convergence curves: {save_path}")

if __name__ == "__main__":
    generate_ablation_barchart()
    generate_convergence_curves()
