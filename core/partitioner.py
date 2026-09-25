import numpy as np
import pandas as pd
from core.data_loader import load_and_validate_dataset
from core.graph_engine import construct_graph_snapshots
from config.logging_config import get_logger

logger = get_logger("Partitioner")

def get_non_iid_subnets(config_path="config/config.yaml"):
    df, scaler, feature_cols, cfg = load_and_validate_dataset(config_path)
    num_nodes = cfg["data"]["num_nodes"]

    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        c1_df = df[df["attack_cat"].isin(["Normal", "Generic", "Exploits", "DoS"])].copy()
        c2_df = df[df["attack_cat"].isin(["Normal", "Fuzzers", "Reconnaissance", "Backdoor", "Analysis", "Shellcode"])].copy()
    else:
        split_point = len(df) // 2
        c1_df = df.iloc[:split_point].copy()
        c2_df = df.iloc[split_point:].copy()

    test_df = df.sample(n=min(5000, len(df)), random_state=cfg["system"]["seed"]).copy()

    logger.info(f"Partitioned Data: Subnet 1 ({len(c1_df):,} flows), Subnet 2 ({len(c2_df):,} flows), Global Test ({len(test_df):,} flows)")

    c1_snaps = construct_graph_snapshots(c1_df.iloc[:4000], scaler, feature_cols, num_nodes)
    c2_snaps = construct_graph_snapshots(c2_df.iloc[:4000], scaler, feature_cols, num_nodes)
    test_snaps = construct_graph_snapshots(test_df, scaler, feature_cols, num_nodes)

    return c1_snaps, c2_snaps, test_snaps

if __name__ == "__main__":
    c1, c2, test = get_non_iid_subnets()
    print(f"? Partitioner Test: Client 1 = {len(c1)} snaps, Client 2 = {len(c2)} snaps, Global Test = {len(test)} snaps.")
