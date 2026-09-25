import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_topology_map(snapshot, y_pred, round_num, save_dir=RESULTS_DIR):
    """Draws 2D network interaction map (Green: Benign Host, Red: Detected Intrusion)."""
    plt.figure(figsize=(9, 7), dpi=300)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(snapshot.num_nodes))
    edges = snapshot.edge_index.t().cpu().numpy()
    G.add_edges_from(edges[:180])
    
    pos = nx.spring_layout(G, seed=42, k=0.3)
    node_colors = ['#2ecc71' if pred == 0 else '#e74c3c' for pred in y_pred]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=280, edgecolors='#2c3e50', linewidths=1.2)
    nx.draw_networkx_edges(G, pos, edge_color='#bdc3c7', alpha=0.35, arrows=True, arrowsize=10)
    
    labels = {i: f"H{i}" for i in range(snapshot.num_nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color='#2c3e50', font_weight='bold')
    
    plt.title(f"Dynamic Host Interaction Topology — Federated Round {round_num}\n(Green: Benign Host | Red: Detected Intrusion Node)", 
              fontsize=11, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"topology_map_round_{round_num}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def plot_publication_dashboard(metrics_history, save_path=os.path.join(RESULTS_DIR, "ieee_performance_dashboard.png")):
    """Builds a 4-quadrant IEEE performance visualization dashboard."""
    rounds = [m['round'] for m in metrics_history]
    losses = [m['loss'] for m in metrics_history]
    f1s = [m['f1_score'] * 100 for m in metrics_history]
    accs = [m['accuracy'] * 100 for m in metrics_history]
    aucs = [m['roc_auc'] * 100 for m in metrics_history]
    fprs = [m['fpr'] * 100 for m in metrics_history]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Convergence Loss
    axs[0, 0].plot(rounds, losses, marker='o', color='#2980b9', linewidth=2.2, markersize=6)
    axs[0, 0].set_title("Federated Convergence Loss (MSE)", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Federated Round")
    axs[0, 0].set_ylabel("Reconstruction Loss")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Accuracy & F1-Score Progression
    axs[0, 1].plot(rounds, accs, marker='s', color='#27ae60', linewidth=2.0, label="Accuracy (%)")
    axs[0, 1].plot(rounds, f1s, marker='^', color='#e67e22', linewidth=2.0, label="F1-Score (%)")
    axs[0, 1].set_title("Detection Accuracy & F1-Score Progression", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("Federated Round")
    axs[0, 1].set_ylabel("Score (%)")
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 3. ROC-AUC vs FPR
    axs[1, 0].plot(rounds, aucs, marker='D', color='#8e44ad', linewidth=2.0, label="ROC-AUC (%)")
    axs[1, 0].plot(rounds, fprs, marker='x', color='#c0392b', linewidth=2.0, linestyle=':', label="FPR (%)")
    axs[1, 0].set_title("ROC-AUC & False Positive Rate (FPR)", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("Federated Round")
    axs[1, 0].set_ylabel("Rate (%)")
    axs[1, 0].legend(loc="center right")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. Summary Table
    axs[1, 1].axis('off')
    table_data = [
        ["Round", "Loss", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        *[
            [
                f"R{m['round']}", 
                f"{m['loss']:.4f}", 
                f"{m['accuracy']*100:.1f}%", 
                f"{m['precision']:.3f}", 
                f"{m['recall']:.3f}", 
                f"{m['f1_score']:.3f}", 
                f"{m['roc_auc']:.3f}"
            ]
            for m in metrics_history
        ]
    ]
    table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    axs[1, 1].set_title("IEEE Empirical Metric Evaluation Summary", fontsize=11, fontweight='bold', pad=10)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def save_metrics_csv(metrics_history, save_path=os.path.join(RESULTS_DIR, "ieee_metrics_results.csv")):
    """Exports metrics history to a structured CSV table."""
    records = []
    for m in metrics_history:
        records.append({
            "Federated_Round": m['round'],
            "Reconstruction_Loss": round(m['loss'], 6),
            "Accuracy_Percent": round(m['accuracy'] * 100, 2),
            "Precision": round(m['precision'], 4),
            "Recall": round(m['recall'], 4),
            "F1_Score": round(m['f1_score'], 4),
            "ROC_AUC": round(m['roc_auc'], 4),
            "False_Positive_Rate": round(m['fpr'], 4)
        })
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    return save_path