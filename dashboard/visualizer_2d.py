import plotly.graph_objects as go
import networkx as nx
import numpy as np

def build_interactive_topology_figure(snapshot, y_pred, recon_error):
    """Constructs a dynamic Plotly 2D topological graph with host telemetry tooltips."""
    G = nx.DiGraph()
    G.add_nodes_from(range(snapshot.num_nodes))
    edges = snapshot.edge_index.t().cpu().numpy()
    G.add_edges_from(edges[:120])

    pos = nx.spring_layout(G, seed=42, k=0.35)
    
    # Edge Traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color='#7f8c8d'),
        hoverinfo='none',
        mode='lines'
    )

    # Node Traces
    node_x = [pos[i][0] for i in range(snapshot.num_nodes)]
    node_y = [pos[i][1] for i in range(snapshot.num_nodes)]
    node_colors = ['#e74c3c' if p == 1 else '#2ecc71' for p in y_pred]
    
    hover_texts = [
        f"Host: 192.168.1.{i+1}<br>Role: {'Attacker / Incursion' if i >= 35 else 'Workstation / Server'}<br>Status: {'🚨 COMPROMISED' if y_pred[i] == 1 else '✅ NORMAL'}<br>Recon MSE: {recon_error[i]:.5f}"
        for i in range(snapshot.num_nodes)
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_texts,
        text=[f"H{i}" for i in range(snapshot.num_nodes)],
        textposition="top center",
        marker=dict(
            color=node_colors,
            size=18,
            line=dict(width=2, color='#2c3e50')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Dynamic Host Topology (Green: Benign | Red: Anomalous Incursion)", font=dict(size=14)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=480,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig
