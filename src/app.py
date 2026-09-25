import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from model import SpatialTemporalGNNAutoencoder

# Page Configuration
st.set_page_config(
    page_title="Fed-GNIDS SOC Monitor",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #262730;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Fed-GNIDS: Autonomous SOC Intrusion Detection Radar")
st.caption("Batch 15 | VEMU Institute of Technology | Guided by Mr. K. Niranjan")

# Sidebar Controls
st.sidebar.header("🕹️ SOC Operational Controls")
model_path = st.sidebar.text_input("Global Checkpoint Path", "data/processed/global_model.pt")
data_path = st.sidebar.text_input("Benchmark Test Data Path", "data/processed/real_test_benchmark.pt")
threshold_sigma = st.sidebar.slider("Anomaly Sensitivity (Sigma Multiplier)", min_value=1.0, max_value=3.5, value=2.0, step=0.1)

sample_limit = st.sidebar.slider("Inspection Window (Snapshots)", min_value=10, max_value=96, value=40, step=5)

# Check Artifact Availability
if not os.path.exists(model_path) or not os.path.exists(data_path):
    st.error("[-] Checkpoint or real test benchmark file missing. Run preprocess.py and server.py first.")
    st.stop()

@st.cache_resource
def load_system(m_path, d_path):
    device = torch.device("cpu")
    model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64)
    model.load_state_dict(torch.load(m_path, weights_only=True))
    model.eval()
    dataset = torch.load(d_path, weights_only=False)
    return model, dataset

model, full_dataset = load_system(model_path, data_path)
dataset = full_dataset[:sample_limit]

# Run Inference
all_scores = []
all_labels = []

with torch.no_grad():
    for seq in dataset:
        target = seq[-1]
        reconstructed, ground_truth = model(seq)
        edge_mse = torch.mean((reconstructed - ground_truth) ** 2, dim=-1)
        all_scores.extend(edge_mse.numpy())
        all_labels.extend(target.y.numpy())

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# Calibrated Threshold
normal_scores = all_scores[all_labels == 0] if 0 in all_labels else all_scores
tau = np.mean(normal_scores) + (threshold_sigma * np.std(normal_scores))
predictions = (all_scores >= tau).astype(int)

# Real Metrics
total_flows = len(all_scores)
flagged_intrusions = int(np.sum(predictions))
normal_flows = total_flows - flagged_intrusions

prec = precision_score(all_labels, predictions, zero_division=0)
rec = recall_score(all_labels, predictions, zero_division=0)
f1 = f1_score(all_labels, predictions, zero_division=0)
auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0

# Top KPI Summary
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Inspected Flows", f"{total_flows:,}")
col2.metric("Flagged Incursions", f"{flagged_intrusions}", delta=f"{flagged_intrusions} Alerts", delta_color="inverse")
col3.metric("Precision", f"{prec * 100:.1f}%")
col4.metric("Recall (Detection Rate)", f"{rec * 100:.1f}%")
col5.metric("Operational Cutoff (τ)", f"{tau:.5f}")

st.divider()

# Center Layout: Real-Time Scatter & Figure Browser
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📊 Dynamic Edge Flow Reconstruction Error (MSE)")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x_axis = np.arange(len(all_scores))
    ax.scatter(x_axis[predictions == 0], all_scores[predictions == 0], color='#2ca02c', alpha=0.5, s=12, label='Normal Flow')
    ax.scatter(x_axis[predictions == 1], all_scores[predictions == 1], color='#d62728', alpha=0.8, s=25, label='Flagged Incursion (APT)')
    ax.axhline(tau, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Threshold (τ = {tau:.4f})')
    ax.set_xlabel("Flow Sequence Index", fontsize=10)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

with right_col:
    st.subheader("📑 Empirical Benchmark Gallery")
    plot_choice = st.selectbox(
        "Select IEEE Verification Plot", 
        ["Baseline Comparison", "Adversarial Robustness", "Ablation Study", "ROC Curve", "Confusion Matrix"]
    )
    
    plot_map = {
        "Baseline Comparison": "results/ieee_baseline_comparison.png",
        "Adversarial Robustness": "results/ieee_adversarial_robustness.png",
        "Ablation Study": "results/ieee_ablation_study.png",
        "ROC Curve": "results/ieee_roc_curve.png",
        "Confusion Matrix": "results/ieee_confusion_matrix.png"
    }

    selected_fig = plot_map.get(plot_choice)
    if os.path.exists(selected_fig):
        st.image(selected_fig, width="stretch")
    else:
        st.info(f"Target figure `{selected_fig}` not found. Run respective script to generate.")

# Live Intrusion Incident Log
st.subheader("🚨 Real-Time Security Incident Stream")
incident_indices = np.where(predictions == 1)[0]

if len(incident_indices) > 0:
    incident_data = {
        "Flow Identifier": [f"FLOW-NB15-{idx:05d}" for idx in incident_indices[:30]],
        "Reconstruction MSE": [round(float(all_scores[idx]), 6) for idx in incident_indices[:30]],
        "Exceedance Ratio": [f"{(all_scores[idx] / tau):.2f}x" for idx in incident_indices[:30]],
        "Ground Truth Telemetry": ["Confirmed Attack" if all_labels[idx] == 1 else "False Alarm" for idx in incident_indices[:30]],
        "Mitigation Action": ["Isolate Communication Edge" for _ in incident_indices[:30]]
    }
    df_incidents = pd.DataFrame(incident_data)
    st.dataframe(df_incidents, width="stretch")
else:
    st.success("No anomalies currently exceed the dynamic operational threshold.")