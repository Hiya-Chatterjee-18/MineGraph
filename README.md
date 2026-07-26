# ⛏️ MineGraph AI: Underground Mine Safety & Risk Intelligence

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/System-Active-success)

**MineGraph AI** is an end-to-end Machine Learning and Graph Neural Network platform designed for real-time underground mine safety monitoring, hazard forecasting, and spatial risk propagation.

---

## 🌟 Key Features

1. **🚦 Real-Time Multi-Sensor Telemetry & Fail-Safe Overrides**
   - Continuously monitors Methane ($CH_4$), Oxygen ($O_2$), Airflow Velocity, Ambient Temperature, Relative Humidity, and Structural Vibration.
   - Includes hard physical limit overrides (e.g., $O_2 \le 16\%$ or $CH_4 \ge 2.0\%$) to guarantee safety enforcement even before statistical ML triggers.

2. **🤖 Multi-Model AI Ensemble Engine**
   - Combines **Logistic Regression**, **Random Forest**, and **Support Vector Machines (SVM)** to generate weighted probability scores ($0\% - 100\%$) and ensemble consensus decisions.

3. **🕸️ Spatial Tunnel Graph Risk Propagation (GNN)**
   - Models the underground mine network as a connected graph $G = (V, E)$ where nodes represent mine shafts/tunnels and edges represent air ventilation ducts.
   - Simulates airborne gas and risk propagation to neighboring mine shafts before hazard physically spreads.

4. **📈 LSTM Time-Series Forecasting**
   - Deep learning sequence modeling predicts Methane trends 60 minutes into the future to alert shift supervisors to impending gas spikes.

5. **🛡️ Automated Emergency Action Protocol Engine**
   - Contextual recommendation generator advising shaft evacuation, auxiliary fan activation, and breaker cutoffs during hazardous events.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[IoT Mine Sensor Telemetry] --> B{Physical Safety Overrides}
    B -->|Breached| C[🚨 CRITICAL OVERRIDE: Evacuate Shaft]
    B -->|Normal Bounds| D[ML Feature Alignment Engine]
    
    D --> E1[Logistic Regression]
    D --> E2[Random Forest]
    D --> E3[Support Vector Machine]
    
    E1 --> F[Ensemble Consensus & Risk Score]
    E2 --> F
    E3 --> F
    
    F --> G[Graph Neural Network Risk Propagation]
    G --> H[LSTM Methane Trend Predictor]
    
    F --> I[Streamlit Dashboard & Action Protocols]
    G --> I
    H --> I
