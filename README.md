
# 🚀 AI-Powered Predictive Maintenance for Industrial Equipment

An end-to-end AI system that predicts equipment failure using real-world time-series data and provides intelligent maintenance recommendations.

---

## 🔗 Live Demo

👉 [https://huggingface.co/spaces/amr-moussa/AI-Powered_Predictive_maintenance](https://huggingface.co/spaces/amr-moussa/AI-Powered_Predictive_maintenance)

---

## 📌 Project Overview

Industrial equipment failures are costly and disruptive.

This project uses **machine learning + deep learning + AI assistants** to:

* Predict **Remaining Useful Life (RUL)**
* Estimate **failure probability**
* Classify **risk level**
* Generate **automated maintenance reports**

All in an interactive web application.

---

## 🧠 Models Used

### 🔹 LSTM (Deep Learning)

* Captures **temporal dependencies**
* Learns degradation patterns over time
* Input: sequences of sensor readings (window = 30 cycles)

---

### 🔹 XGBoost (Machine Learning)

* Uses **engineered features**:

  * mean
  * std
  * min / max
  * last value
  * trend (slope)
* Fast and highly interpretable

---

### 🔁 Model Comparison

Users can switch between models in real-time to compare predictions.

---

## 📊 Dataset

* **NASA C-MAPSS (FD001)**
* Simulated turbofan engine degradation
* Each engine runs until failure

### Features:

* 3 operational settings
* 21 sensors (subset used after feature selection)
* Time-series data

---

## ⚙️ System Architecture

```text
Raw Sensor Data (NASA format)
        ↓
Preprocessing + Scaling
        ↓
Sequence Creation (30 cycles)
        ↓
   ┌───────────────┐
   │               │
 LSTM          XGBoost
   │               │
   └──────┬────────┘
          ↓
     Predicted RUL
          ↓
 Failure Probability
          ↓
   Risk Classification
          ↓
 AI Maintenance Report (LLM)
```

---

## 🤖 AI Report Generation

The system integrates LLMs to generate human-readable reports:

* 🧠 **Gemini API (primary)**
* ⚡ **Groq (fallback)**
* 🛡️ Rule-based fallback if APIs fail

Reports include:

* Condition summary
* Risk explanation
* Recommended actions
* Urgency level

---

## 🖥️ Features

* Upload **raw NASA dataset (no preprocessing required)**
* Predict:

  * RUL
  * Failure probability
  * Risk level
* Choose model: **LSTM vs XGBoost**
* Generate **AI-powered maintenance reports**
* Download report as `.txt`
* Fully deployed on **Hugging Face Spaces**

---

## 📷 Demo

*(Add screenshots from your `/Images` folder here)*

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch** (LSTM)
* **XGBoost**
* **Scikit-learn**
* **Streamlit**
* **Google Gemini API**
* **Groq API**
* **Hugging Face Spaces**

---

## 📂 Project Structure

```text
Dataset/
Assets/
  FD001/
    model files

notebooks/
   EDA+Preprocessing.ipynb
   LSTM.ipynb
   XGBoost.ipynb

app.py              # Streamlit app
.env
llm.py              # LLM integration 
requirements.txt
README.md
```

---

## 🚀 Installation (Local)

```bash
git clone https://github.com/Amr1Moussa/AI-Powered-Predictive-Maintenance-for-Industrial-Equipment
cd AI-Powered-Predictive-Maintenance-for-Industrial-Equipment

pip install -r requirements.txt

streamlit run app.py
```

---

## 🔑 Environment Variables

Create `.env` file:

```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

---

## 💡 Real-World Application

This system can be adapted to:

* DC motors (current, vibration, temperature)
* Industrial pumps
* Manufacturing machines
* Predictive maintenance in IoT systems

---

## 📈 Future Improvements

* Real-time IoT streaming
* Transformer-based models
* Mobile app integration
* Edge deployment (ESP32 + sensors)


## ⭐ Key Takeaway

This project demonstrates how to move from:

> raw sensor data → AI model → actionable decision → deployed product

---
