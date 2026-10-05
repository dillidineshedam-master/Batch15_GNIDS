import time
import torch
import numpy as np
import pandas as pd
from hybrid_pipeline import DualEngineFedGNIDS

device = torch.device("cpu")
print("[*] Initializing Real-Time NIDS Ingress Pipeline on CPU...")

# 1. Load test data and calibrated checkpoint
test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)
model = DualEngineFedGNIDS(50, 16, 64).to(device)
model.load_state_dict(torch.load("results/checkpoints/hybrid_dual_engine_gnids.pt", weights_only=False))
model.eval()

# Calibrated production threshold
THETA = 0.4050

# 2. Warm up CPU cache to avoid cold-start timing skew
warmup_seq = [s.to(device) for s in test_data[0]]
for _ in range(10):
    with torch.no_grad():
        _ = model(warmup_seq)

print("[+] Warmup complete. Streaming live test flow windows...\n")
print(f"{'Time Window':<14} | {'Flows':<7} | {'Latency (ms)':<13} | {'Flows/sec':<10} | {'Window Acc':<11} | {'Cumulative Acc'}")
print("-" * 80)

latencies = []
total_flows = 0
correct_predictions = 0
window_correct = 0
window_flows = 0
window_idx = 1

window_step = max(1, len(test_data) // 10)  # Stream in 10 telemetry windows

with torch.no_grad():
    for idx, seq in enumerate(test_data, 1):
        seq = [s.to(device) for s in seq]
        
        t_start = time.perf_counter()
        _, logits, _, gt_labels = model(seq)
        probs = torch.sigmoid(logits)
        t_elapsed = time.perf_counter() - t_start
        
        preds = (probs >= THETA).long()
        labels = gt_labels.long()
        
        n_edges = labels.size(0)
        correct_in_snapshot = (preds == labels).sum().item()
        
        latencies.append((t_elapsed / n_edges) * 1e6)  # microseconds per flow
        total_flows += n_edges
        correct_predictions += correct_in_snapshot
        
        window_flows += n_edges
        window_correct += correct_in_snapshot
        
        if idx % window_step == 0 or idx == len(test_data):
            win_acc = (window_correct / window_flows) * 100
            cum_acc = (correct_predictions / total_flows) * 100
            batch_latency_ms = t_elapsed * 1000
            throughput = n_edges / (t_elapsed + 1e-9)
            
            print(f"Window {window_idx:02d}/10   | {window_flows:<7} | {batch_latency_ms:6.2f} ms     | {throughput:8.1f} | {win_acc:6.2f}%    | {cum_acc:6.2f}%")
            
            window_correct = 0
            window_flows = 0
            window_idx += 1

print("-" * 80)
print("\n" + "="*50)
print("     REAL-TIME HARDWARE EXECUTION PROFILE")
print("="*50)
print(f"  Total Processed Flows     : {total_flows:,}")
print(f"  Overall Real-Time Accuracy: {correct_predictions / total_flows * 100:.2f}%")
print(f"  Mean Per-Flow Latency     : {np.mean(latencies):.2f} us")
print(f"  P95 Per-Flow Latency      : {np.percentile(latencies, 95):.2f} us")
print(f"  Sustained Line Throughput : {1e6 / np.mean(latencies):,.0f} flows/sec")
print("="*50)