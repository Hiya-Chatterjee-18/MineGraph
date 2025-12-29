import streamlit as st
import pandas as pd
try:
    import joblib
except ModuleNotFoundError:
    import sklearn.externals.joblib as joblib


from MLModelPipeline import combined_ml_safety_decision

# ======================================================
# PAGE CONFIG & TITLE
# ======================================================
st.set_page_config(layout="wide")
st.title("MineGraph: your mine safety guide", text_alignment="center")
st.subheader("Your AI-powered mine safety guide", text_alignment="center")

# ======================================================
# SECTION 1: CURRENT TUNNEL SAFETY (ML)
# ======================================================
st.header("🔹 Know about a particular tunnel (Current Condition)")

tunnel_id = st.selectbox("Select Tunnel", ["A", "B", "C"])

st.subheader("Enter current sensor readings")

col1, col2 = st.columns(2)

with col1:
    methane_pct = st.number_input("Methane (%)", 0.0)
    temperature_c = st.number_input("Temperature (°C)", 0.0)
    humidity_pct = st.number_input("Humidity (%)", 0.0)

with col2:
    airflow_mps = st.number_input("Airflow (m/s)", 0.0)
    vibration_mm_s = st.number_input("Vibration (mm/s)", 0.0)
    oxygen_pct = st.number_input("Oxygen (%)", 0.0)

# --------------------------------------------------
# Prepare input for ML models
# --------------------------------------------------
current_tunnel_data = pd.DataFrame([{
    "methane_pct": methane_pct,
    "temperature_c": temperature_c,
    "humidity_pct": humidity_pct,
    "airflow_mps": airflow_mps,
    "vibration_mm_s": vibration_mm_s,
    "oxygen_pct": oxygen_pct,
    "methane_avg": methane_pct,
    "methane_change": 0.0,
    "risk_score": methane_pct * 20
}])

# ======================================================
# RUN ML MODELS
# ======================================================
if st.button("Check Current Safety Status"):

    result = combined_ml_safety_decision(current_tunnel_data)

    decision_map = {"Safe": 0, "Not Safe": 1}
    model_decisions = result["model_wise_decision"]

    lr_val = decision_map[model_decisions["Logistic Regression"]]
    rf_val = decision_map[model_decisions["Random Forest"]]
    svm_val = decision_map[model_decisions["SVM"]]

    # --------------------------------------------------
    # HARD SAFETY OVERRIDE (DEFINE FIRST)
    # --------------------------------------------------
    hard_risk = False
    if oxygen_pct <= 10:
        hard_risk = True
        st.error("🚨 CRITICAL: Oxygen level dangerously low")

    # --------------------------------------------------
    # FAIL-SAFE OVERALL DECISION (WITH CLEAR SAFE GUARD)
    # --------------------------------------------------
    st.subheader("🚦 Current Tunnel Safety Status")

    clearly_safe = (
        methane_pct <= 1.2 and
        temperature_c <= 35 and
        airflow_mps >= 2.0 and
        vibration_mm_s <= 2.0 and
        oxygen_pct >= 20.0
    )

    if hard_risk:
        current_tunnel_status = "Unsafe"
        st.error("❌ NOT SAFE — critical physical limit breached")

    elif clearly_safe:
        current_tunnel_status = "Safe"
        st.success("✅ SAFE — sensor readings within normal operating range")

    elif lr_val == 1 or rf_val == 1 or svm_val == 1:
        current_tunnel_status = "Unsafe"
        st.error("❌ NOT SAFE — ML risk detected")

    else:
        current_tunnel_status = "Safe"
        st.success("✅ SAFE — no significant risk detected")

    # ==================================================
    # MODEL-WISE OUTPUT
    # ==================================================
    st.subheader("📊 Model-wise Decisions")

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("### Logistic Regression")
        st.bar_chart(pd.DataFrame({"Risk Level": [lr_val]}))
        if lr_val == 0:
            st.success("Safe")
            st.caption("Risk probability below safety threshold")
        else:
            st.error("Flags Risk")
            st.caption("Risk probability exceeds safety threshold")

    with colB:
        st.markdown("### Random Forest")
        st.bar_chart(pd.DataFrame({"Risk Level": [rf_val]}))
        if rf_val == 0:
            st.success("Safe")
            st.caption("Most decision trees agree conditions are normal")
        else:
            st.error("Flags Risk")
            st.caption("Unsafe sensor pattern detected")

    with colC:
        st.markdown("### SVM")
        st.bar_chart(pd.DataFrame({"Risk Level": [svm_val]}))
        if svm_val == 0:
            st.success("Safe")
            st.caption("Within learned safe boundary")
        else:
            st.error("Flags Risk")
            st.caption("Outside learned safe boundary")

    st.markdown("---")

    # ======================================================
    # SECTION 2: PAST & FUTURE CONDITION (LSTM)
    # ======================================================
    st.header("🔹 Past & Future Condition")

    ts_df = joblib.load("models/preprocessed_timeseries.pkl")
    lstm_df = joblib.load("models/lstm_output.pkl")

    tunnel_history = (
        ts_df[ts_df["tunnel_id"] == tunnel_id]
        .sort_values("timestamp")
        .tail(10)
    )

    future_methane = lstm_df[
        lstm_df["tunnel_id"] == tunnel_id
    ]["predicted_methane"].values[0]

    trend = lstm_df[
        lstm_df["tunnel_id"] == tunnel_id
    ]["trend"].values[0]

    st.write(f"**Trend:** {trend}")

    past_df = tunnel_history[["timestamp", "methane_pct"]]
    future_df = pd.DataFrame({
        "timestamp": ["Future"],
        "methane_pct": [future_methane]
    })

    plot_df = pd.concat([past_df, future_df], ignore_index=True)
    st.line_chart(plot_df.set_index("timestamp")["methane_pct"])

    st.markdown("---")

    # ======================================================
    # SECTION 3: DYNAMIC GNN RISK PROPAGATION
    # ======================================================
    st.header("🔹 Overall Mine Safety Overview")

    base_risk = {"A": "Safe", "B": "Safe", "C": "Safe"}
    base_risk[tunnel_id] = current_tunnel_status

    connections = {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B"]
    }

    gnn_risk = base_risk.copy()

    for tunnel, neighbors in connections.items():
        if base_risk[tunnel] == "Unsafe":
            for nb in neighbors:
                gnn_risk[nb] = "Warning"

    gnn_df = pd.DataFrame({
        "Tunnel": gnn_risk.keys(),
        "GNN Risk Output": gnn_risk.values()
    })

    st.table(gnn_df)

    st.info(
        "Risk propagates from unsafe tunnels to connected tunnels, "
        "allowing early warning before danger physically spreads."
    )

st.markdown("---")
st.caption("To all coal miners: Your courage, hard work, and resilience power the world from deep underground. Every shift you take is a sacrifice for your families and for society. May safety always come first, and may you return home healthy every day. You are seen, valued, and deeply respected. 🖤⛏️")
