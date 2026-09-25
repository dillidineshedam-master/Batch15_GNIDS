import os
import time
import torch
import numpy as np
import pandas as pd
from model import SpatialTemporalGNNAutoencoder

def profile_system():
    print("="*75)
    print("      FED-GNIDS EDGE DEPLOYMENT & COMPUTATIONAL PROFILER")
    print("="*75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Target Device: {device}")

    # 1. Parameter Footprint
    model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0

    print(f"\n[+] --- Architectural Footprint ---")
    print(f"    Total Parameters        : {total_params:,}")
    print(f"    Trainable Parameters    : {trainable_params:,}")
    print(f"    Model Weights Size (RAM): {model_size_kb:.2f} KB ({model_size_kb/1024:.3f} MB)")

    # 2. FL Communication Bandwidth Profiling
    # In each round, client sends updated parameters (float32 array)
    comm_payload_bytes = sum(p.numel() * 4 for p in model.parameters())
    comm_payload_kb = comm_payload_bytes / 1024.0
    print(f"\n[+] --- Federated Network Transmission ---")
    print(f"    Uplink Payload / Round  : {comm_payload_kb:.2f} KB")
    print(f"    Downlink Payload / Round: {comm_payload_kb:.2f} KB")
    print(f"    Total per Client (8 Rnd): {(comm_payload_kb * 2 * 8) / 1024:.2f} MB")

    # 3. Inference Latency Benchmark
    test_path = "data/processed/real_test_benchmark.pt"
    if not os.path.exists(test_path):
        print(f"[-] Benchmark test data not found at {test_path}.")
        return

    test_data = torch.load(test_path, weights_only=False)
    model.eval()

    latencies_ms = []
    # Warmup
    for seq in test_data[:5]:
        seq = [s.to(device) for s in seq]
        with torch.no_grad():
            _ = model(seq)

    # Benchmark run
    with torch.no_grad():
        for seq in test_data:
            seq = [s.to(device) for s in seq]
            t0 = time.perf_counter()
            _ = model(seq)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies_ms)
    std_lat = np.std(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    
    # 250 flows per snapshot
    flows_per_snapshot = 250
    throughput_fps = (flows_per_snapshot / (avg_lat / 1000.0))

    print(f"\n[+] --- Real-Time Edge Latency Benchmark ({len(test_data)} sequences) ---")
    print(f"    Average Latency / Window: {avg_lat:.2f} ms (+/- {std_lat:.2f} ms)")
    print(f"    95th Percentile Latency : {p95_lat:.2f} ms")
    print(f"    Min / Max Latency       : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"    Per-Flow Processing Time: {(avg_lat / flows_per_snapshot) * 1000:.2f} microseconds")
    print(f"    Inference Throughput    : {throughput_fps:,.0f} flows / second")

    # Save profiler results to CSV
    profile_df = pd.DataFrame([{
        "Total Parameters": total_params,
        "Model Size (KB)": round(model_size_kb, 2),
        "Payload per Round (KB)": round(comm_payload_kb, 2),
        "Avg Window Latency (ms)": round(avg_lat, 2),
        "95th Pct Latency (ms)": round(p95_lat, 2),
        "Throughput (flows/sec)": round(throughput_fps, 0)
    }])
    os.makedirs("results", exist_ok=True)
    profile_df.to_csv("results/system_profiling.csv", index=False)
    print(f"\n[+] Profiling report saved to results/system_profiling.csv")
    print("="*75)

if __name__ == "__main__":
    profile_system()