import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated.server_aggregator import FederatedServerManager

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set IEEE-compatible plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.autolayout": True
})

# -------------------------------------------------------------
# Figure 1: Out-of-Sample Reconstruction Error Distribution
# -------------------------------------------------------------
def plot_anomaly_separation():
    mgr = FederatedServerManager("config/config.yaml")
    ckpt_path = "results/checkpoints/best_global_gnids.pt"
    mgr.global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    mgr.global_model.eval()

    all_snaps = mgr.test_snapshots
    calib_snaps = all_snaps[:5]
    test_snaps = all_snaps[5:10]

    with torch.no_grad():
        recon_c = mgr.global_model(calib_snaps)
        target_c = torch.stack([s.x for s in calib_snaps], dim=1)
        feat_err_c = torch.mean((recon_c - target_c) ** 2, dim=1).cpu().numpy()
        y_c = (torch.stack([s.y for s in calib_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

        benign = (y_c == 0)
        std_feat = np.std(feat_err_c[benign], axis=0) + 1e-6
        node_err_c = np.mean(feat_err_c / std_feat, axis=1)

        mu = np.mean(node_err_c[benign])
        sig = np.std(node_err_c[benign])
        tau = float(mu + 2.5 * sig)

        recon_t = mgr.global_model(test_snaps)
        target_t = torch.stack([s.x for s in test_snaps], dim=1)
        feat_err_t = torch.mean((recon_t - target_t) ** 2, dim=1).cpu().numpy()
        node_err_t = np.mean(feat_err_t / std_feat, axis=1)
        y_t = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

    benign_scores = node_err_t[y_t == 0]
    attack_scores = node_err_t[y_t == 1]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    ax.scatter(np.where(y_t == 0)[0], benign_scores, color="#2b5c8f", label="Benign Hosts (TN)", alpha=0.85, edgecolors="none", s=35)
    ax.scatter(np.where(y_t == 1)[0], attack_scores, color="#d95f02", label="Intrusion Hosts (TP)", alpha=0.9, marker="^", s=50)
    ax.axhline(tau, color="#e41a1c", linestyle="--", linewidth=1.5, label=f"Decision Threshold (tau = {tau:.2f})")

    ax.set_xlabel("Host Node Identifier Index")
    ax.set_ylabel("Normalized Reconstruction Error")
    ax.set_title("Temporal Anomaly Separation (k = 2.5 sigma)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig_png = os.path.join(FIG_DIR, "anomaly_separation.png")
    fig_pdf = os.path.join(FIG_DIR, "anomaly_separation.pdf")
    fig.savefig(fig_png)
    fig.savefig(fig_pdf)
    plt.close()
    print(f"[PlotEngine] Anomaly separation vector plots saved: {fig_png}, {fig_pdf}")

# -------------------------------------------------------------
# Figure 2: Federated Training Convergence History
# -------------------------------------------------------------
def plot_convergence():
    metrics_path = os.path.join(RESULTS_DIR, "metrics", "federated_training_metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"[PlotEngine] No metrics file found at {metrics_path}. Skipping convergence plot.")
        return

    df = pd.read_csv(metrics_path)
    df.columns = df.columns.str.strip().str.lower()

    fig, ax1 = plt.subplots(figsize=(6, 3.5), dpi=300)

    # Primary axis: Reconstruction Loss
    color_loss = "#1b9e77"
    ax1.set_xlabel("Federated Communication Round")
    ax1.set_ylabel("Global Reconstruction Loss (MSE)", color=color_loss)
    line1 = ax1.plot(df["round"], df["loss"], marker="o", markersize=4, color=color_loss, linewidth=1.8, label="Global Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis: F1-Score Progression
    ax2 = ax1.twinx()
    color_f1 = "#7570b3"
    ax2.set_ylabel("Validation F1-Score", color=color_f1)
    line2 = ax2.plot(df["round"], df["f1_score"], marker="s", markersize=4, color=color_f1, linewidth=1.8, linestyle="--", label="F1-Score")
    ax2.tick_params(axis="y", labelcolor=color_f1)
    ax2.set_ylim(0.7, 1.05)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    ax1.set_title("Fed-GNIDS Convergence Across Communication Rounds")

    fig_png = os.path.join(FIG_DIR, "federated_convergence.png")
    fig_pdf = os.path.join(FIG_DIR, "federated_convergence.pdf")
    fig.savefig(fig_png)
    fig.savefig(fig_pdf)
    plt.close()
    print(f"[PlotEngine] Convergence curve plots saved: {fig_png}, {fig_pdf}")

# -------------------------------------------------------------
# LaTeX Table Export
# -------------------------------------------------------------
def export_latex_table():
    csv_path = os.path.join(RESULTS_DIR, "multiseed_statistical_results.csv")
    if not os.path.exists(csv_path):
        print(f"[PlotEngine] No statistical CSV found at {csv_path}. Skipping LaTeX export.")
        return
    df = pd.read_csv(csv_path)
    latex_code = df.to_latex(
        index=False, 
        caption="Comparative Performance of Fed-GNIDS Against Baselines (Mean $\\pm$ Standard Deviation across Seeds)", 
        label="tab:benchmark_results"
    )
    tex_path = os.path.join(RESULTS_DIR, "ieee_results_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"[PlotEngine] IEEE LaTeX table formatted and saved: {tex_path}")

if __name__ == "__main__":
    plot_anomaly_separation()
    plot_convergence()
    export_latex_table()