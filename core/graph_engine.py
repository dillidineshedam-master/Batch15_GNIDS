import numpy as np
import torch
from torch_geometric.data import Data
from config.logging_config import get_logger

logger = get_logger("GraphEngine")

def construct_graph_snapshots(df_partition, scaler, feature_cols, num_nodes=50, window_size=500):
    scaled_matrix = scaler.transform(df_partition[feature_cols].values)

    df = df_partition.copy().reset_index(drop=True)
    np.random.seed(42)

    # Subnet Mapping:
    # Hosts 0-14  : Enterprise Core Servers
    # Hosts 15-34 : Benign Workstations
    # Hosts 35-49 : External / Attacker Hosts
    src_nodes = []
    dst_nodes = []

    for idx, row in df.iterrows():
        if int(row["label"]) == 1:
            u = np.random.randint(35, num_nodes) # Attacker source
            v = np.random.randint(0, 15)         # Target server
        else:
            u = np.random.randint(15, 35)        # Workstation source
            v = np.random.randint(0, 15)         # Server destination
        src_nodes.append(u)
        dst_nodes.append(v)

    df["src"] = src_nodes
    df["dst"] = dst_nodes
    df["window_id"] = np.arange(len(df)) // window_size

    snapshot_ids = sorted(df["window_id"].unique())
    snapshots = []

    for wid in snapshot_ids:
        sub_df = df[df["window_id"] == wid]
        if len(sub_df) < 20:
            continue

        edge_list = []
        edge_attr_list = []
        node_features_sum = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        node_counts = np.zeros(num_nodes, dtype=np.float32)
        node_labels = np.zeros(num_nodes, dtype=int)

        for idx_row, row in sub_df.iterrows():
            u, v = int(row["src"]), int(row["dst"])
            edge_list.append([u, v])

            flow_feat = scaled_matrix[idx_row]
            edge_attr_list.append(flow_feat[:3])

            node_features_sum[u] += flow_feat
            node_features_sum[v] += flow_feat * 0.5
            node_counts[u] += 1
            node_counts[v] += 1

            if int(row["label"]) == 1:
                node_labels[u] = 1 # Mark attacker host

        node_counts[node_counts == 0] = 1.0
        node_features = node_features_sum / node_counts[:, None]

        edge_index = torch.tensor(np.array(edge_list), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)
        snapshots.append(snapshot)

    logger.info(f"Built {len(snapshots)} temporal graph snapshots across {num_nodes} hosts.")
    return snapshots
