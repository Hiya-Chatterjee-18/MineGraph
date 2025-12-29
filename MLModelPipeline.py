# MLModelPipeline.py

import joblib
import pandas as pd
from collections import Counter

# Load trained models
lr_model = joblib.load("models/lr_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def combined_ml_safety_decision(current_tunnel_data: pd.DataFrame):

    # Align features
    current_tunnel_data = current_tunnel_data[feature_columns]

    lr_pred = lr_model.predict(current_tunnel_data)[0]
    rf_pred = rf_model.predict(current_tunnel_data)[0]
    svm_pred = svm_model.predict(current_tunnel_data)[0]

    lr_label = label_encoder.inverse_transform([lr_pred])[0]
    rf_label = label_encoder.inverse_transform([rf_pred])[0]
    svm_label = label_encoder.inverse_transform([svm_pred])[0]

    def safe_or_not(label):
        return "Safe" if label == "Safe" else "Not Safe"

    decisions = {
        "Logistic Regression": safe_or_not(lr_label),
        "Random Forest": safe_or_not(rf_label),
        "SVM": safe_or_not(svm_label)
    }

    final_decision = Counter(decisions.values()).most_common(1)[0][0]

    return {
        "model_wise_decision": decisions,
        "final_decision": final_decision
    }
