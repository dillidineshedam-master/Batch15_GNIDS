import flwr as fl
import torch
import numpy as np
import importlib
import os

preprocess_mod = importlib.import_module("01_preprocess")
model_mod = importlib.import_module("02_model")
metrics_mod = importlib.import_module("03_metrics_viz")

extract_dynamic_snapshots = preprocess_mod.extract_dynamic_snapshots
GNIDS_Autoencoder = model_mod.GNIDS_Autoencoder
evaluate_intrusion_detection = model_mod.evaluate_intrusion_detection

plot_topology_map = metrics_mod.plot_topology_map
plot_publication_dashboard = metrics_mod.plot_publication_dashboard
save_metrics_csv = metrics_mod.save_metrics_csv

metrics_history = []
eval_snapshots, ip_map = extract_dynamic_snapshots()
global_model = GNIDS_Autoencoder()

def evaluate_fn(server_round: int, parameters, config):
    if server_round == 0:
        return 0.0, {}

    params_dict = zip(global_model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    global_model.load_state_dict(state_dict, strict=True)

    metrics = evaluate_intrusion_detection(global_model, eval_snapshots)
    current_loss = float(np.mean(metrics['recon_error']))

    print("\n" + "=" * 60)
    print(f"📈 [SERVER ROUND {server_round} SUMMARY]")
    print(f"   Reconstruction Loss : {current_loss:.6f}")
    print(f"   Accuracy            : {metrics['accuracy'] * 100:.2f}%")
    print(f"   Precision           : {metrics['precision']:.4f}")
    print(f"   Recall              : {metrics['recall']:.4f}")
    print(f"   F1-Score            : {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC             : {metrics['roc_auc']:.4f}")
    print(f"   False Positive Rate : {metrics['fpr']:.4f}")
    print("=" * 60 + "\n")

    saved_top_path = plot_topology_map(eval_snapshots[0], metrics['y_pred'], server_round)
    print(f"🖼️ Saved Graphical Network Topology Map to '{saved_top_path}'")

    metrics_history.append({
        "round": server_round,
        "loss": current_loss,
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1_score": metrics['f1_score'],
        "roc_auc": metrics['roc_auc'],
        "fpr": metrics['fpr']
    })

    return current_loss, {"f1_score": metrics['f1_score'], "accuracy": metrics['accuracy']}

if __name__ == "__main__":
    print("=" * 60)
    print("🌐 MODULE 4: CENTRAL FEDERATED AGGREGATOR SERVER & ACS ENGINE")
    print("=" * 60)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate_fn
    )

    print("🚀 Central Aggregator Server listening on 0.0.0.0:8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

    dash_path = plot_publication_dashboard(metrics_history)
    csv_path = save_metrics_csv(metrics_history)

    print("\n" + "=" * 60)
    print(f"📊 Saved Publication Dashboard to '{dash_path}'")
    print(f"📑 Saved IEEE Metrics CSV Table to '{csv_path}'")
    print("🏆 PROJECT DEMO COMPLETE: All graphs, visual maps, and CSV tables stored in 'results/' folder!")
    print("=" * 60)