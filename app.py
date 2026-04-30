# app.py

import os
import time
import streamlit as st
import pandas as pd
import torch
import joblib
import json
import numpy as np
import torch.nn as nn
from google import genai
from groq import Groq

# load environment variables from .env file if exists
from dotenv import load_dotenv
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FD001_DIR = os.path.join(BASE_DIR, "Dataset", "FD001")


def sequence_to_features(X):
    mean_features = X.mean(axis=1)
    std_features = X.std(axis=1)
    min_features = X.min(axis=1)
    max_features = X.max(axis=1)
    last_features = X[:, -1, :]

    time = np.arange(X.shape[1])
    slopes = []

    for sample in X:
        sample_slopes = []
        for sensor_idx in range(sample.shape[1]):
            slope = np.polyfit(time, sample[:, sensor_idx], 1)[0]
            sample_slopes.append(slope)
        slopes.append(sample_slopes)

    slope_features = np.array(slopes)

    return np.concatenate([
        mean_features,
        std_features,
        min_features,
        max_features,
        last_features,
        slope_features
    ], axis=1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.regressor(last_out).squeeze(1)


def compute_failure_probability(predicted_rul, rul_max):
    return float(np.clip(1 - (predicted_rul / rul_max), 0, 1))



def suggest_action(predicted_rul, failure_probability):
    if failure_probability < 0.35 and predicted_rul > 80:
        return "Healthy", "Continue normal operation. Recheck during the next scheduled inspection."
    elif failure_probability < 0.65 and predicted_rul > 40:
        return "Warning", "Schedule maintenance soon. Increase monitoring frequency and inspect sensor trends."
    elif failure_probability < 0.85 and predicted_rul > 15:
        return "High Risk", "Plan maintenance urgently. Prepare spare parts and reduce operating load if possible."
    else:
        return "Critical", "Immediate inspection required. Consider shutdown to prevent failure."



def generate_llm_report(predicted_rul, p_failure, risk, action):
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    prompt = f"""
You are an industrial predictive maintenance assistant.

Machine status:
- Predicted RUL: {predicted_rul:.2f} cycles
- Failure Probability: {p_failure * 100:.1f}%
- Risk Level: {risk}
- Suggested Action: {action}

Write a professional maintenance report with:
1. Condition summary
2. Risk explanation (not just restating the risk level)
3. Maintenance recommendation
4. Urgency level

Do not change the numbers.
Do not invent sensor values.
Do not include any meta information or disclaimers. Only provide the report content.
"""

    # ---------------- GEMINI FIRST ----------------
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)

            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return response.text

        except Exception as e:
            print("Gemini failed, switching to Groq...", e)

    # ---------------- GROQ FALLBACK ----------------
    if groq_key:
        try:
            client = Groq(api_key=groq_key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  
                messages=[
                    {"role": "system", "content": "You are an industrial predictive maintenance assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            print("Groq also failed...", e)

    # ---------------- FINAL FALLBACK ----------------
    return f"""
### Maintenance Report

**Condition Summary:**  
The machine is classified as **{risk}**.

**Predicted RUL:** {predicted_rul:.2f} cycles  
**Failure Probability:** {p_failure * 100:.1f}%  

**Recommended Action:**  
{action}
"""


@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(FD001_DIR, "config.json")
    scaler_path = os.path.join(FD001_DIR, "scaler_fd001.pkl")
    lstm_path = os.path.join(FD001_DIR, "best_lstm_fd001_regularized.pt")
    xgb_path = os.path.join(FD001_DIR, "xgb_model.pkl")

    with open(config_path, "r") as f:
        config = json.load(f)

    scaler = joblib.load(scaler_path)

    lstm_model = LSTMRegressor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    xgb_model = joblib.load(xgb_path)
    
    return lstm_model, xgb_model, scaler, config, device

def predict_XGBoost(raw_df, xgb_model, scaler, config):
    feature_cols = config["feature_cols"]
    window_size = config["window_size"]
    rul_max = config["rul_max"]

    seq = raw_df[feature_cols].values

    if len(seq) >= window_size:
        seq = seq[-window_size:]
    else:
        pad_len = window_size - len(seq)
        pad = np.repeat(seq[:1], pad_len, axis=0)
        seq = np.vstack([pad, seq])

    seq_scaled = scaler.transform(seq)

    # XGBoost expects: (samples, window, features)
    x_seq = np.expand_dims(seq_scaled, axis=0)

    x_features = sequence_to_features(x_seq)

    predicted_rul = float(xgb_model.predict(x_features)[0])
    predicted_rul = float(np.clip(predicted_rul, 0, rul_max))

    p_failure = compute_failure_probability(predicted_rul, rul_max)
    risk, action = suggest_action(predicted_rul, p_failure)

    return predicted_rul, p_failure, risk, action


def predict_LSTM(raw_df, model, scaler, config, device):
    feature_cols = config["feature_cols"]
    window_size = config["window_size"]
    rul_max = config["rul_max"]

    seq = raw_df[feature_cols].values

    if len(seq) >= window_size:
        seq = seq[-window_size:]
    else:
        pad_len = window_size - len(seq)
        pad = np.repeat(seq[:1], pad_len, axis=0)
        seq = np.vstack([pad, seq])

    seq_scaled = scaler.transform(seq)
    x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_rul = model(x).item()

    predicted_rul = float(np.clip(predicted_rul, 0, rul_max))
    p_failure = compute_failure_probability(predicted_rul, rul_max)
    risk, action = suggest_action(predicted_rul, p_failure)

    return predicted_rul, p_failure, risk, action



# Streamlit App
st.title("AI-Powered Predictive Maintenance for Industrial Equipment")
st.write("Upload a raw NASA C-MAPSS file (no headers, 26 columns).")

uploaded_file = st.file_uploader(
    "Upload raw NASA C-MAPSS file",
    type=["txt", "csv"]
)      


lstm_model, xgb_model, scaler, config, device = load_assets()


def prepare_uploaded_data(raw_df, config):
    feature_cols = config["feature_cols"]

    if raw_df.shape[1] != 26:
        raise ValueError(f"Expected 26 columns, got {raw_df.shape[1]}")


    nasa_cols = ['unit', 'cycle'] + \
                ['op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                [f'sensor_{i}' for i in range(1, 22)]

    raw_df = raw_df.iloc[:, :26].copy()
    raw_df.columns = nasa_cols

    input_df = raw_df[feature_cols].copy()

    return input_df


# Main prediction flow
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

        input_df = prepare_uploaded_data(raw_df, config)

        st.success("Raw NASA C-MAPSS format detected and processed.")

        #st.subheader("Processed Sensor Input")
        #st.dataframe(input_df.head())
        
        model_choice = st.selectbox("Choose prediction model",["LSTM", "XGBoost"])
        
        if model_choice == "LSTM":
            predicted_rul, p_failure, risk, action = predict_LSTM(input_df, lstm_model, scaler, config, device)
        else:
            predicted_rul, p_failure, risk, action = predict_XGBoost(input_df, xgb_model, scaler, config)


        st.subheader("Prediction Results")
        st.metric("Predicted RUL", f"{predicted_rul:.2f} cycles")
        st.metric("Failure Probability", f"{p_failure * 100:.1f}%")
        st.metric("Risk Level", risk)

        st.subheader("Suggested Action")
        st.write(action)
        
        
        if st.button("Generate AI Maintenance Report"):
            with st.spinner("Generating maintenance report..."):
                report = generate_llm_report(
                    predicted_rul,
                    p_failure,
                    risk,
                    action
                )

            st.subheader("Maintenance Report")
            st.write(report)
        
            st.download_button(
                    "Download Report",
                    report,
                    file_name="maintenance_report.txt"
                )    
            
        st.subheader("Model Insight")
        st.write("Prediction is based on recent degradation trends in sensor readings over last 30 cycles.")

    except Exception as e:
        st.error(f"Input processing failed: {e}")
        

