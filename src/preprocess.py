import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

NUMERICAL_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 
    'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 
    'sinpkt', 'dinpkt', 'smean', 'dmean'
]

def clean_and_map_df(df, num_hosts=50, scaler=None, is_train=True):
    available_features = [f for f in NUMERICAL_FEATURES if f in df.columns]
    
    # Hash deterministic communicating endpoints
    src_hash = df['proto'].astype(str) + df['state'].astype(str)
    dst_hash = df['service'].astype(str) + df['sttl'].astype(str)
    
    df['src_host'] = (src_hash.apply(hash).abs() % num_hosts).astype(int)
    df['dst_host'] = (dst_hash.apply(hash).abs() % num_hosts).astype(int)
    
    mask = df['src_host'] == df['dst_host']
    df.loc[mask, 'dst_host'] = (df.loc[mask, 'dst_host'] + 1) % num_hosts

    if is_train:
        scaler = MinMaxScaler()
        df[available_features] = scaler.fit_transform(df[available_features].fillna(0))
    else:
        df[available_features] = scaler.transform(df[available_features].fillna(0))

    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    df['label'] = df[label_col].astype(int)

    return df, available_features, scaler

def build_temporal_graph_snapshots(df, feature_cols, snapshot_size=250, time_window_T=5, num_hosts=50, max_snapshots=300):
    total_possible = len(df) // snapshot_size
    num_snapshots = min(total_possible, max_snapshots)
    snapshots = []

    for s_idx in range(num_snapshots):
        batch = df.iloc[s_idx * snapshot_size : (s_idx + 1) * snapshot_size]
        
        edge_index = []
        edge_attr = []
        edge_labels = []

        for _, row in batch.iterrows():
            u = int(row['src_host'])
            v = int(row['dst_host'])
            attrs = row[feature_cols].values.astype(np.float32)
            lbl = int(row['label'])
            
            edge_index.append([u, v])
            edge_attr.append(attrs)
            edge_labels.append(lbl)

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(np.array(edge_attr), dtype=torch.float32)
        edge_labels_tensor = torch.tensor(edge_labels, dtype=torch.long)
        x_tensor = torch.eye(num_hosts, dtype=torch.float32)

        data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=edge_labels_tensor,
            num_nodes=num_hosts
        )
        snapshots.append(data)

    sequences = []
    for i in range(len(snapshots) - time_window_T + 1):
        seq = snapshots[i : i + time_window_T]
        sequences.append(seq)

    return sequences

def partition_non_iid(sequences, num_clients=2, alpha=0.5):
    np.random.seed(42)
    num_samples = len(sequences)
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients), size=1)[0]
    counts = (proportions * num_samples).astype(int)
    counts[-1] = num_samples - np.sum(counts[:-1])

    client_splits = {}
    start = 0
    for c_id in range(num_clients):
        end = start + counts[c_id]
        client_splits[c_id] = sequences[start:end]
        print(f"    -> Client {c_id}: Assigned {len(client_splits[c_id])} sequences")
        start = end

    return client_splits

def main():
    train_path = "data/raw/UNSW_NB15_training-set.csv"
    test_path = "data/raw/UNSW_NB15_testing-set.csv"
    save_dir = "data/processed"
    os.makedirs(save_dir, exist_ok=True)

    print("[*] --- Processing Real Training Set ---")
    df_train = pd.read_csv(train_path)
    df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_train, features, scaler = clean_and_map_df(df_train, is_train=True)
    train_sequences = build_temporal_graph_snapshots(df_train, features, max_snapshots=300)
    client_splits = partition_non_iid(train_sequences, num_clients=2, alpha=0.5)

    for c_id, seqs in client_splits.items():
        save_file = os.path.join(save_dir, f"client_{c_id}_data.pt")
        torch.save(seqs, save_file)
        print(f"[+] Saved {save_file}")

    print("\n[*] --- Processing Real Testing Set (Official Benchmark with Attacks) ---")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        # Shuffle to mix attack flows and benign flows uniformly across snapshot windows
        df_test = df_test.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_test, _, _ = clean_and_map_df(df_test, scaler=scaler, is_train=False)
        test_sequences = build_temporal_graph_snapshots(df_test, features, max_snapshots=100)
        
        total_attacks = sum([(seq[-1].y == 1).sum().item() for seq in test_sequences])
        total_normal = sum([(seq[-1].y == 0).sum().item() for seq in test_sequences])
        print(f"[+] Ground-Truth Test Slice: {total_normal:,} Normal Flows, {total_attacks:,} Attack Flows")

        test_save_file = os.path.join(save_dir, "real_test_benchmark.pt")
        torch.save(test_sequences, test_save_file)
        print(f"[+] Saved Benchmark Evaluation Set: {test_save_file}")
    else:
        print(f"[!] Warning: {test_path} not found.")

    print("\n[SUCCESS] Preprocessing completed.")

if __name__ == "__main__":
    main()