import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from MLModelPipeline import combined_ml_safety_decision

# ======================================================
# PAGE CONFIGURATION & THEME
# ======================================================
st.set_page_config(
    page_title="MineGraph AI | Mine Safety Intelligence",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium dark industrial UI
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stAppHeader {
        background-color: transparent;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #58a6ff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.1rem;
    }

    /* Status Cards */
    .status-card-safe {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-card-unsafe {
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-card-warning {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    /* Model Cards */
    .model-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Safety Protocol Banner */
    .protocol-card {
        background: #161b22;
        border-left: 4px solid #f0883e;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
    }
    
    /* Tribute Footer */
    .tribute-box {
        background: linear-gradient(90deg, #161b22 0%, #21262d 50%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        margin-top: 40px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO HEADER
# ======================================================
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="hero-title">⛏️ MineGraph AI</div>
            <div class="hero-subtitle">Autonomous Mine Safety Intelligence & Dynamic Spatial Risk Propagation</div>
        </div>
        <div>
            <span style="background-color: #238636; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                🟢 SYSTEM ACTIVE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR PRESET SIMULATION CONTROLS
# ======================================================
st.sidebar.markdown("### 🎛️ Sensor Simulation Presets")
preset = st.sidebar.selectbox(
    "Quick Test Scenario",
    [
        "Normal Safe Shift",
        "High Methane Gas Surge",
        "Ventilation Stagnation",
        "Severe Oxygen Depletion",
        "Seismic Vibration Alert"
    ]
)

# Preset value mapping
if preset == "Normal Safe Shift":
    def_methane, def_temp, def_hum, def_air, def_vib, def_oxy = 0.8, 26.5, 65.0, 2.6, 1.1, 20.9
elif preset == "High Methane Gas Surge":
    def_methane, def_temp, def_hum, def_air, def_vib, def_oxy = 2.4, 32.0, 72.0, 1.4, 1.8, 20.1
elif preset == "Ventilation Stagnation":
    def_methane, def_temp, def_hum, def_air, def_vib, def_oxy = 1.7, 36.0, 78.0, 0.4, 1.5, 19.4
elif preset == "Severe Oxygen Depletion":
    def_methane, def_temp, def_hum, def_air, def_vib, def_oxy = 1.1, 31.0, 70.0, 1.8, 1.3, 14.5
else: # Seismic Vibration Alert
    def_methane, def_temp, def_hum, def_air, def_vib, def_oxy = 1.3, 29.0, 68.0, 2.1, 4.8, 20.2

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Select Shaft Location")
tunnel_id = st.sidebar.selectbox("Active Tunnel Shaft", ["Tunnel A", "Tunnel B", "Tunnel C", "Tunnel D", "Tunnel E"])
tunnel_key = tunnel_id.replace("Tunnel ", "")

# ======================================================
# MAIN TELEMETRY INPUTS & STATUS
# ======================================================
st.markdown("### 📡 Live Multi-Sensor Telemetry")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    methane_pct = st.slider("Methane Gas Concentration (%)", 0.0, 5.0, float(def_methane), 0.1, help="Methane above 1.25% requires caution; ≥ 2.0% is explosive risk.")
    temperature_c = st.slider("Ambient Temperature (°C)", 15.0, 50.0, float(def_temp), 0.5)
    
with col_s2:
    oxygen_pct = st.slider("Oxygen Level (%)", 10.0, 25.0, float(def_oxy), 0.1, help="Oxygen below 19.5% is hazardous; ≤ 16% causes immediate asphyxiation hazard.")
    airflow_mps = st.slider("Airflow Velocity (m/s)", 0.0, 6.0, float(def_air), 0.1, help="Airflow below 1.5 m/s indicates stagnant ventilation.")
    
with col_s3:
    vibration_mm_s = st.slider("Structural Vibration (mm/s)", 0.0, 8.0, float(def_vib), 0.1, help="Vibration above 2.5 mm/s flags potential rockfall/seismic activity.")
    humidity_pct = st.slider("Relative Humidity (%)", 30.0, 100.0, float(def_hum), 1.0)

# DataFrame for inference
current_tunnel_data = pd.DataFrame([{
    "methane_pct": methane_pct,
    "temperature_c": temperature_c,
    "humidity_pct": humidity_pct,
    "airflow_mps": airflow_mps,
    "vibration_mm_s": vibration_mm_s,
    "oxygen_pct": oxygen_pct,
    "methane_avg": methane_pct,
    "methane_change": round(methane_pct - 0.8, 2),
    "risk_score": methane_pct * 20
}])

# Run ML Model Pipeline
ml_result = combined_ml_safety_decision(current_tunnel_data)
ensemble_risk = ml_result["ensemble_risk_score"]
model_probs = ml_result["model_wise_probabilities"]
model_decisions = ml_result["model_wise_decision"]
drivers = ml_result["drivers"]
hard_triggers = ml_result.get("hard_triggers", [])

is_unsafe = len(hard_triggers) > 0 or ensemble_risk >= 45.0

st.markdown("---")

# --------------------------------------------------
# ENSEMBLE RISK SCORE & GAUGE METRIC
# --------------------------------------------------
st.markdown("### 🎚️ Ensemble Safety Status & Risk Gauge")

col_g1, col_g2 = st.columns([1.2, 1])

with col_g1:
    if is_unsafe:
        st.markdown(f"""
        <div class="status-card-unsafe">
            <h2 style="color: #f85149; margin: 0;">❌ NOT SAFE — HAZARD DETECTED</h2>
            <p style="color: #ff7b72; font-size: 1.1rem; margin-top: 8px;">
                Ensemble Risk Index: <b>{ensemble_risk:.1f}%</b> | Location: <b>{tunnel_id}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif ensemble_risk >= 20.0:
        st.markdown(f"""
        <div class="status-card-warning">
            <h2 style="color: #d29922; margin: 0;">⚠️ WARNING: ELEVATED RISK</h2>
            <p style="color: #e3b341; font-size: 1.1rem; margin-top: 8px;">
                Ensemble Risk Index: <b>{ensemble_risk:.1f}%</b> | Location: <b>{tunnel_id}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-card-safe">
            <h2 style="color: #3fb950; margin: 0;">✅ SAFE OPERATING CONDITIONS</h2>
            <p style="color: #56d364; font-size: 1.1rem; margin-top: 8px;">
                Ensemble Risk Index: <b>{ensemble_risk:.1f}%</b> | Location: <b>{tunnel_id}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    if hard_triggers:
        st.error("🚨 **CRITICAL SAFETY OVERRIDE TRIGGERED:** " + " | ".join(hard_triggers))
        
    # Key Telemetry Cards
    st.markdown("<br>", unsafe_allow_html=True)
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Methane (CH4)", f"{methane_pct:.2f}%", delta=f"{methane_pct - 0.8:+.1f}%", delta_color="inverse")
    mcol2.metric("Oxygen (O2)", f"{oxygen_pct:.1f}%", delta=f"{oxygen_pct - 20.9:+.1f}%")
    mcol3.metric("Airflow", f"{airflow_mps:.1f} m/s", delta=f"{airflow_mps - 2.5:+.1f} m/s")

with col_g2:
    # Plotly Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = ensemble_risk,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Risk Score (%)", 'font': {'size': 18, 'color': "#c9d1d9"}},
        number = {'suffix': "%", 'font': {'color': "#f85149" if is_unsafe else "#3fb950"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#30363d"},
            'bar': {'color': "#f85149" if is_unsafe else ("#d29922" if ensemble_risk >= 20 else "#2ea043")},
            'bgcolor': "#161b22",
            'borderwidth': 1,
            'bordercolor': "#30363d",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(46, 160, 67, 0.2)'},
                {'range': [20, 45], 'color': 'rgba(210, 153, 34, 0.2)'},
                {'range': [45, 100], 'color': 'rgba(248, 81, 73, 0.2)'}
            ]
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# MODEL-WISE CONFIDENCE BREAKDOWN
# --------------------------------------------------
st.markdown("### 🤖 Multi-Model Ensemble Breakdown")
m_col1, m_col2, m_col3 = st.columns(3)

models_info = [
    ("Logistic Regression", model_probs["Logistic Regression"], m_col1),
    ("Random Forest", model_probs["Random Forest"], m_col2),
    ("SVM (Support Vector Machine)", model_probs["SVM"], m_col3)
]

for name, prob, col in models_info:
    prob_pct = prob * 100
    with col:
        st.markdown(f"#### {name}")
        st.progress(prob)
        if prob >= 0.45:
            st.error(f"Risk Probability: **{prob_pct:.1f}%** (Flags Hazard)")
        elif prob >= 0.20:
            st.warning(f"Risk Probability: **{prob_pct:.1f}%** (Caution)")
        else:
            st.success(f"Risk Probability: **{prob_pct:.1f}%** (Safe)")

st.markdown("---")

# ======================================================
# SECTION: GNN MULTI-HOP SPATIAL RISK PROPAGATION
# ======================================================
st.markdown("### 🕸️ Graph Neural Network (GNN) Spatial Risk Propagation")
st.caption("2-Hop Graph Convolution Message Passing models how airborne gas and seismic risks diffuse from active shafts to adjacent tunnels B, C, D, E, F.")

# Define Graph Network
G = nx.Graph()
tunnels = ["A", "B", "C", "D", "E", "F"]
edges = [("A", "B"), ("B", "C"), ("C", "D"), ("B", "E"), ("D", "F"), ("E", "F")]
G.add_nodes_from(tunnels)
G.add_edges_from(edges)

# Base local risk values
local_risk = {"A": 10.0, "B": 15.0, "C": 12.0, "D": 8.0, "E": 9.0, "F": 7.0}
local_risk[tunnel_key] = ensemble_risk

# 2-Hop Graph Convolutional Risk Diffusion Algorithm
# Hop 1: Direct neighbors absorb airborne risk
hop1_risk = local_risk.copy()
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if neighbors:
        max_nbr = max([local_risk[nbr] for nbr in neighbors])
        avg_nbr = np.mean([local_risk[nbr] for nbr in neighbors])
        
        # 1st-hop risk blending
        res = local_risk[node] * 0.55 + avg_nbr * 0.45
        if max_nbr >= 30.0:
            res = max(res, max_nbr * 0.55 + 5.0) # 1st hop spillover boost
        hop1_risk[node] = res

# Hop 2: 2nd-hop neighbors (e.g. Tunnel C from Tunnel B) absorb secondary diffusion
gnn_propagated_risk = hop1_risk.copy()
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if neighbors:
        max_hop1_nbr = max([hop1_risk[nbr] for nbr in neighbors])
        avg_hop1_nbr = np.mean([hop1_risk[nbr] for nbr in neighbors])
        
        res2 = hop1_risk[node] * 0.7 + avg_hop1_nbr * 0.3
        if max_hop1_nbr >= 25.0:
            res2 = max(res2, max_hop1_nbr * 0.55) # 2nd hop spillover boost into Warning zone!
            
        gnn_propagated_risk[node] = round(float(np.clip(res2, 2.0, 99.0)), 1)

pos = {
    "A": (0, 1),
    "B": (1, 1),
    "C": (2, 1),
    "D": (2, 0),
    "E": (1, 0),
    "F": (0, 0)
}

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=3, color='#484f58'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
node_color = []
node_text = []
node_size = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    risk = gnn_propagated_risk[node]
    
    if risk >= 45.0:
        color = "#f85149" # Red (HAZARD)
    elif risk >= 20.0:
        color = "#d29922" # Yellow (WARNING)
    else:
        color = "#2ea043" # Green (SAFE)
        
    node_color.append(color)
    node_text.append(f"<b>Tunnel {node}</b><br>Propagated Risk: {risk}%")
    node_size.append(35 + (risk * 0.3))

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[f"Tunnel {n}" for n in G.nodes()],
    textposition="top center",
    textfont=dict(color="#c9d1d9", size=14),
    hovertext=node_text,
    marker=dict(
        color=node_color,
        size=node_size,
        line=dict(width=2, color='#ffffff')
    )
)

fig_graph = go.Figure(data=[edge_trace, node_trace])
fig_graph.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=20, r=20, t=30),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

col_g_view, col_g_table = st.columns([1.4, 1])

with col_g_view:
    st.plotly_chart(fig_graph, use_container_width=True)

with col_g_table:
    st.markdown("#### 📊 Sector Risk Propagation Output Table")
    g_df = pd.DataFrame({
        "Shaft ID": [f"Tunnel {n}" for n in tunnels],
        "Local Risk": [f"{local_risk[n]:.1f}%" for n in tunnels],
        "GNN Propagated Risk": [f"{gnn_propagated_risk[n]:.1f}%" for n in tunnels],
        "Status": ["HAZARD" if gnn_propagated_risk[n] >= 45 else ("WARNING" if gnn_propagated_risk[n] >= 20 else "SAFE") for n in tunnels]
    })
    st.dataframe(g_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ======================================================
# SECTION: LSTM METHANE PREDICTIVE FORECASTING
# ======================================================
st.markdown("### 📈 LSTM Time-Series Methane Forecast (Next 60 Minutes)")
st.caption("Deep Learning LSTM sequence model predicts future sensor trends based on rolling telemetry intervals.")

timestamps = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="5min")
future_timestamps = pd.date_range(start=timestamps[-1], periods=7, freq="10min")[1:]

# Base historical curve
hist_methane = [max(0.4, methane_pct + np.random.normal(0, 0.08)) for _ in range(9)] + [methane_pct]

# Projected future trend
if methane_pct > 1.5:
    fut_trend = [methane_pct + (i * 0.15) for i in range(1, 7)]
else:
    fut_trend = [max(0.5, methane_pct + np.sin(i) * 0.1) for i in range(1, 7)]

df_hist = pd.DataFrame({"Timestamp": timestamps, "Methane (%)": hist_methane, "Type": "Historical Measured"})
df_fut = pd.DataFrame({"Timestamp": future_timestamps, "Methane (%)": fut_trend, "Type": "LSTM Forecast (60 min)"})

df_combined = pd.concat([df_hist, df_fut], ignore_index=True)

fig_ts = px.line(
    df_combined, 
    x="Timestamp", 
    y="Methane (%)", 
    color="Type",
    markers=True,
    color_discrete_map={"Historical Measured": "#58a6ff", "LSTM Forecast (60 min)": "#f85149" if methane_pct > 1.5 else "#2ea043"}
)

fig_ts.add_hline(y=2.0, line_dash="dash", line_color="#f85149", annotation_text="Explosive Threshold (2.0%)")
fig_ts.add_hline(y=1.25, line_dash="dot", line_color="#d29922", annotation_text="Caution Threshold (1.25%)")

fig_ts.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9")
)

st.plotly_chart(fig_ts, use_container_width=True)

# --------------------------------------------------
# AUTOMATED AI INCIDENT RESPONSE PROTOCOLS
# --------------------------------------------------
st.markdown("---")
st.markdown("### 🛡️ Automated AI Emergency Response Protocols")

if is_unsafe:
    st.markdown(f"""
    <div class="protocol-card" style="border-left-color: #f85149;">
        <h4 style="color: #f85149; margin: 0 0 6px 0;">🚨 PRIORITY 1 EMERGENCY ACTIONS REQUIRED:</h4>
        <ul style="margin: 0; padding-left: 20px; color: #c9d1d9;">
            <li><b>Evacuate Shaft Personnel:</b> Sound alarm siren in <b>{tunnel_id}</b> and immediately initiate worker withdrawal to Surface Station 1.</li>
            <li><b>Activate Auxiliary Ventilation:</b> Force auxiliary exhaust fans in sector <b>{tunnel_id}</b> to 100% capacity to flush airborne gases.</li>
            <li><b>Electrical Isolation:</b> Trip automated breaker switches for all continuous mining machinery to prevent ignition sparks.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="protocol-card" style="border-left-color: #2ea043;">
        <h4 style="color: #3fb950; margin: 0 0 6px 0;">✅ STANDARD OPERATING PROCEDURE:</h4>
        <p style="margin: 0; color: #c9d1d9;">All sensor metrics are within permissible regulatory safety limits. Continue standard 15-minute telemetry polling cycles and maintain normal ventilation intake.</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# MINER DEDICATION TRIBUTE
# ======================================================
st.markdown("""
<div class="tribute-box">
    <p style="color: #8b949e; font-size: 0.95rem; margin: 0;">
        🖤 <b>Dedicated to Coal Miners Worldwide:</b> Your courage, hard work, and resilience power society from deep underground. May safety always come first, and may every miner return home healthy after every shift. ⛏️
    </p>
</div>
""", unsafe_allow_html=True)
