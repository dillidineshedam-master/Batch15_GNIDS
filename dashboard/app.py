import sys
import os

# Guarantee project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import yaml

from core.partitioner import get_non_iid_subnets
from models.autoencoder import SpatialTemporalAutoencoder
from models.threshold_engine import compute_anomaly_metrics
from dashboard.visualizer_2d import build_interactive_topology_figure

st.set_page_config(
    page_title="GNIDS - Federated Intrusion Platform",
    page_icon="🛡️",
    layout="wide"
)

# Load Configuration & Model Checkpoint
@st.cache_resource
def load_system():
    cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    _, _, test_snapshots = get_non_iid_subnets(cfg_path)
    
    model = SpatialTemporalAutoencoder(
        in_features=cfg["model"]["in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["heads"]
    )
    
    ckpt_path = os.path.join(PROJECT_ROOT, "results", "checkpoints", "best_global_gnids.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        model.eval()
        
    return cfg, test_snapshots, model

cfg, test_snapshots, model = load_system()

st.title("🛡️ Federated Spatial-Temporal Graph NIDS Live Operations Center")
st.markdown("**Real-Time Enterprise Telemetry Stream, Dynamic Graph Anomaly Isolation & Federated State**")

# Sidebar Controls
st.sidebar.header("⚙️ Telemetry Controls")
auto_stream = st.sidebar.checkbox("Enable Live 10-Min Snapshot Stream", value=True)
stream_speed = st.sidebar.slider("Snapshot Step Duration (seconds)", min_value=1, max_value=8, value=3)

if "snap_idx" not in st.session_state:
    st.session_state.snap_idx = 0

current_idx = st.session_state.snap_idx % len(test_snapshots)
current_seq = test_snapshots[max(0, current_idx - 7) : current_idx + 1]
if len(current_seq) < 8:
    current_seq = test_snapshots[:8]

k_multiplier = cfg["detection"]["threshold_sigma_multiplier"]
metrics = compute_anomaly_metrics(model, current_seq, k_multiplier=k_multiplier)

# KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
detected_count = int(np.sum(metrics["y_pred"]))

kpi1.metric("Active Monitored Hosts", f"{test_snapshots[0].num_nodes} Hosts")
kpi2.metric("Telemetry Window", f"Window #{current_idx + 1} (10-Min Slice)")
kpi3.metric("Detected Threats", f"{detected_count} Hosts", delta=f"{detected_count} incursion(s)", delta_color="inverse")
kpi4.metric("Empirical ROC-AUC", f"{metrics['roc_auc']:.4f}")

# Main Layout
col_graph, col_table = st.columns([3, 2])

with col_graph:
    st.subheader(f"🌐 Dynamic Network Topology Map (Snapshot {current_idx + 1})")
    fig = build_interactive_topology_figure(current_seq[-1], metrics["y_pred"], metrics["recon_error"])
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("📋 Streaming Host Anomaly Table")
    table_records = []
    for i in range(test_snapshots[0].num_nodes):
        status = "🚨 INTRUSION" if metrics["y_pred"][i] == 1 else "✅ NORMAL"
        table_records.append({
            "Host IP": f"192.168.1.{i+1}",
            "Subnet Role": "Attacker Source" if i >= 35 else "Enterprise Host",
            "Reconstruction MSE": round(float(metrics["recon_error"][i]), 5),
            "Detection Status": status
        })
    df_table = pd.DataFrame(table_records).sort_values(by="Reconstruction MSE", ascending=False).reset_index(drop=True)
    st.dataframe(df_table, height=440, use_container_width=True)

# Performance & Federated Training Metrics
st.markdown("---")
st.subheader("📊 Global Federated Training Progression")
metrics_csv = os.path.join(PROJECT_ROOT, "results", "metrics", "federated_training_metrics.csv")
if os.path.exists(metrics_csv):
    fed_df = pd.read_csv(metrics_csv)
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(fed_df.set_index("round")[["loss"]], height=240)
    with c2:
        st.line_chart(fed_df.set_index("round")[["accuracy", "roc_auc", "f1_score"]], height=240)

if auto_stream:
    time.sleep(stream_speed)
    st.session_state.snap_idx += 1
    st.rerun()
