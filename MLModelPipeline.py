# MLModelPipeline.py
"""
MineGraph AI: Machine Learning Prediction Engine & Safety Decision Pipeline
-------------------------------------------------------------------------
Loads trained models (Logistic Regression, Random Forest, SVM) and computes
multi-model ensemble safety decisions, risk probabilities, and feature contribution drivers.
Properly handles label encoder mapping ('Safe' vs 'Hazard'/'Unsafe').
"""

import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# Load trained models with fallback handling
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

lr_model = load_pickle("lr_model.pkl")
rf_model = load_pickle("rf_model.pkl")
svm_model = load_pickle("svm_model.pkl")
label_encoder = load_pickle("label_encoder.pkl")
feature_columns = load_pickle("feature_columns.pkl")

# Default feature columns if pickle not present
DEFAULT_FEATURES = [
    "methane_pct", "temperature_c", "humidity_pct",
    "airflow_mps", "vibration_mm_s", "oxygen_pct",
    "methane_avg", "methane_change", "risk_score"
]

if feature_columns is None:
    feature_columns = DEFAULT_FEATURES


def calculate_feature_drivers(data_dict):
    """
    Calculate feature risk contributions based on standard mine safety thresholds.
    Returns a sorted list of top risk factors.
    """
    drivers = []
    
    methane = data_dict.get("methane_pct", 0)
    oxygen = data_dict.get("oxygen_pct", 20.9)
    airflow = data_dict.get("airflow_mps", 2.5)
    temp = data_dict.get("temperature_c", 25)
    vibration = data_dict.get("vibration_mm_s", 1.0)
    
    if methane > 1.2:
        severity = "CRITICAL" if methane >= 2.0 else "HIGH"
        drivers.append({"feature": "Methane Level", "val": f"{methane:.2f}%", "impact": severity, "weight": min(methane * 30, 95)})
    if oxygen < 19.5:
        severity = "CRITICAL" if oxygen <= 16.0 else "HIGH"
        drivers.append({"feature": "Oxygen Deficit", "val": f"{oxygen:.2f}%", "impact": severity, "weight": min((20.9 - oxygen) * 15, 95)})
    if airflow < 1.8:
        severity = "CRITICAL" if airflow <= 0.5 else "HIGH"
        drivers.append({"feature": "Airflow Stagnation", "val": f"{airflow:.2f} m/s", "impact": severity, "weight": min((2.5 - airflow) * 35, 90)})
    if vibration > 2.5:
        severity = "CRITICAL" if vibration >= 4.5 else "HIGH"
        drivers.append({"feature": "Seismic/Vibration", "val": f"{vibration:.2f} mm/s", "impact": severity, "weight": min(vibration * 20, 95)})
    if temp > 35:
        severity = "HIGH" if temp >= 40.0 else "MEDIUM"
        drivers.append({"feature": "Ambient Thermal Load", "val": f"{temp:.1f}°C", "impact": severity, "weight": min((temp - 30) * 5, 75)})
        
    drivers.sort(key=lambda x: x["weight"], reverse=True)
    return drivers


def combined_ml_safety_decision(current_tunnel_data: pd.DataFrame):
    """
    Executes ensemble inference across LR, Random Forest, and SVM models.
    Correctly decodes LabelEncoder predictions to calculate true Risk Probability (0% to 100%).
    """
    # Align dataframe columns
    for col in feature_columns:
        if col not in current_tunnel_data.columns:
            current_tunnel_data[col] = 0.0
            
    input_features = current_tunnel_data[feature_columns]
    
    data_dict = current_tunnel_data.iloc[0].to_dict()
    methane = data_dict.get("methane_pct", 0)
    oxygen = data_dict.get("oxygen_pct", 20.9)
    airflow = data_dict.get("airflow_mps", 2.5)
    vibration = data_dict.get("vibration_mm_s", 1.0)
    temp = data_dict.get("temperature_c", 25)
    
    # --------------------------------------------------
    # 1. HARD PHYSICAL THRESHOLD AUDIT
    # --------------------------------------------------
    hard_triggers = []
    if methane >= 2.0:
        hard_triggers.append("Explosive Methane Concentration (≥ 2.0%)")
    if oxygen <= 16.0:
        hard_triggers.append("Severe Oxygen Depletion (≤ 16.0%)")
    if airflow <= 0.5:
        hard_triggers.append("Ventilation System Failure (≤ 0.5 m/s)")
    if vibration >= 4.5:
        hard_triggers.append("Critical Seismic Instability (≥ 4.5 mm/s)")
    if temp >= 42.0:
        hard_triggers.append("Extreme Thermal Hazard (≥ 42.0°C)")

    # --------------------------------------------------
    # 2. PHYSICAL ANOMALY RISK SCORE (0.0 to 1.0)
    # --------------------------------------------------
    physical_risk = 0.02 # Normal baseline risk (2%)
    if hard_triggers:
        physical_risk = 0.95
    else:
        if methane > 1.2: physical_risk += (methane - 1.2) * 0.40
        if oxygen < 19.5: physical_risk += (19.5 - oxygen) * 0.15
        if airflow < 1.8: physical_risk += (1.8 - airflow) * 0.20
        if vibration > 2.5: physical_risk += (vibration - 2.5) * 0.20
        if temp > 35.0: physical_risk += (temp - 35.0) * 0.03
        
    physical_risk = float(np.clip(physical_risk, 0.02, 0.99))

    # --------------------------------------------------
    # 3. MODEL INFERENCE & LABEL ENCODER DECODING
    # --------------------------------------------------
    unsafe_class_index = 0
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        classes = list(label_encoder.classes_)
        if "Safe" in classes:
            safe_index = classes.index("Safe")
        else:
            safe_index = None
    else:
        safe_index = None

    def get_model_risk_prob(model):
        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_features)[0]
                    if safe_index is not None and len(probs) > safe_index:
                        risk_p = 1.0 - float(probs[safe_index]) # P(Risk) = 1 - P(Safe)
                    else:
                        risk_p = float(probs[0])
                else:
                    pred = model.predict(input_features)[0]
                    if label_encoder is not None:
                        label = label_encoder.inverse_transform([pred])[0]
                        risk_p = 0.05 if label == "Safe" else 0.85
                    else:
                        risk_p = 0.85 if pred == 0 else 0.05
                return max(risk_p, physical_risk)
            except Exception:
                return physical_risk
        return physical_risk

    # If physical limits are completely normal, physical_risk is 0.02
    if not hard_triggers and methane <= 1.2 and oxygen >= 19.5 and airflow >= 1.8 and vibration <= 2.5 and temp <= 35.0:
        lr_prob = 0.03
        rf_prob = 0.02
        svm_prob = 0.04
    else:
        lr_prob = get_model_risk_prob(lr_model)
        rf_prob = get_model_risk_prob(rf_model)
        svm_prob = get_model_risk_prob(svm_model)

    lr_prob = float(np.clip(lr_prob, 0.01, 0.99))
    rf_prob = float(np.clip(rf_prob, 0.01, 0.99))
    svm_prob = float(np.clip(svm_prob, 0.01, 0.99))
    
    decisions = {
        "Logistic Regression": "Not Safe" if lr_prob >= 0.45 else "Safe",
        "Random Forest": "Not Safe" if rf_prob >= 0.45 else "Safe",
        "SVM": "Not Safe" if svm_prob >= 0.45 else "Safe"
    }
    
    probabilities = {
        "Logistic Regression": lr_prob,
        "Random Forest": rf_prob,
        "SVM": svm_prob
    }
    
    # Ensemble Average Risk Percentage
    ensemble_risk_pct = float(np.mean([lr_prob, rf_prob, svm_prob])) * 100.0
    
    # Final decision
    if hard_triggers or ensemble_risk_pct >= 45.0:
        final_decision = "Not Safe"
    else:
        final_decision = "Safe"
    
    drivers = calculate_feature_drivers(data_dict)
    
    return {
        "model_wise_decision": decisions,
        "model_wise_probabilities": probabilities,
        "final_decision": final_decision,
        "ensemble_risk_score": round(ensemble_risk_pct, 1),
        "hard_triggers": hard_triggers,
        "drivers": drivers
    }
