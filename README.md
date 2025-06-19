# -isro-fault-detector

# ISRO Satellite Fault and Anomaly Detection System

A real-time AI-powered system for detecting anomalies and classifying faults in satellite telemetry data using both unsupervised and supervised machine learning models. Designed to meet standards suitable for academic, research, and institutional deployment.

---

## Project Overview

This system focuses on two critical problems in satellite data monitoring:

1. **Anomaly Detection**: Identifying unusual patterns in telemetry data that may signal faults or out-of-norm behavior.
2. **Fault Classification**: Predicting known fault types based on historical labeled data.

The solution uses:
- **Isolation Forest** (unsupervised model) for anomaly detection
- **XGBoost Classifier** (supervised model) for fault prediction
- **SHAP Explainability** for transparent model insights
- **Streamlit** to enable interactive, browser-based usage

---

## Key Features

- Real-time CSV data upload and processing
- Anomaly detection with user-defined threshold slider
- Fault classification using trained model
- SHAP-based feature importance explanation (via training cell)
- Downloadable results in CSV format
- Compatible with deployment on Streamlit Cloud

---

## Dataset

The project uses a public telemetry dataset available at:

[Download dataset.csv](https://zenodo.org/record/12588359/files/dataset.csv)

You can also upload your own telemetry data (in CSV format) for custom analysis.

---

## Requirements

Install all dependencies using the following file:

`requirements.txt`

