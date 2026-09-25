import os
import sys
import time
import torch
import numpy as np

from core.data_loader import load_and_validate_dataset
from core.live_sniffer import LiveFlowAggregator
from models.autoencoder import SpatialTemporalAutoencoder
from config.logging_config import get_logger

logger = get_logger("LiveDetector")

def main():
    print("=" * 70)
    print("🛡️ LAUNCHING GNIDS LIVE REAL-TIME PACKET INTRUSION MONITOR")
    print("=" * 70)

    _, scaler, _, cfg = load_and_validate_dataset("config/config.yaml")

    model = SpatialTemporalAutoencoder(
        in_features=cfg["model"]["in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["heads"]
    )

    ckpt_path = "results/checkpoints/best_global_gnids.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        logger.info(f"Loaded global trained weights from '{ckpt_path}'")
    else:
        logger.warning("No checkpoint found! Operating with baseline parameters.")

    model.eval()
    aggregator = LiveFlowAggregator(num_nodes=cfg["data"]["num_nodes"], window_duration=3.0)
    snapshot_buffer = []

    print("\n[*] Monitoring streaming host telemetry (Press Ctrl+C to terminate)...\n")

    try:
        step = 1
        while True:
            time.sleep(3.0)
            snapshot = aggregator.build_snapshot_from_live_flows(scaler)
            snapshot_buffer.append(snapshot)
            if len(snapshot_buffer) > 8:
                snapshot_buffer.pop(0)

            if len(snapshot_buffer) == 8:
                with torch.no_grad():
                    recon = model(snapshot_buffer)
                    target = torch.stack([s.x for s in snapshot_buffer], dim=1)
                    mse_per_host = torch.mean((recon - target) ** 2, dim=[1, 2]).numpy()

                    threshold = np.quantile(mse_per_host, 0.70)
                    anomalies = np.where(mse_per_host >= threshold)[0]

                    print(f"[{time.strftime('%H:%M:%S')}] Snapshot #{step:03d} | "
                          f"Active Edges: {snapshot.edge_index.shape[1]:02d} | "
                          f"Threshold MSE: {threshold:.5f} | "
                          f"🚨 Flagged Incursions ({len(anomalies)}): {anomalies[:5].tolist()}...")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Buffering initial snapshots ({len(snapshot_buffer)}/8)...")

            step += 1

    except KeyboardInterrupt:
        print("\n[+] Live monitor terminated gracefully.")

if __name__ == "__main__":
    main()
