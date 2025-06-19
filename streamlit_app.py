import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ISRO Fault Detection System",
    layout="centered"
)

st.title("ISRO Satellite Fault and Anomaly Detection")

st.markdown("""
This web-based AI system performs:
- Unsupervised anomaly detection using Isolation Forest
- Supervised fault classification using XGBoost
- Real-time analysis and CSV download for further study
""")

try:
    scaler = joblib.load("scaler.pkl")
    iso_model = joblib.load("iso_model.pkl")
    clf = joblib.load("xgb_model.pkl")
    xgb_available = True
except Exception as e:
    xgb_available = False
    st.error("Model files could not be loaded. Please check your deployment folder.")

uploaded_file = st.file_uploader("Upload telemetry data in CSV format", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file).dropna()
        df_numeric = df.select_dtypes(include=[np.number])
        st.success(f"File loaded successfully. Data shape: {df.shape}")

        df_scaled = scaler.transform(df_numeric)

        scores = iso_model.decision_function(df_scaled)
        preds = iso_model.predict(df_scaled)

        df['Anomaly_Score'] = scores
        df['Anomaly_Flag'] = preds

        st.subheader("Anomaly Detection Results")
        st.dataframe(df[['Anomaly_Score', 'Anomaly_Flag']].head(10))

        st.markdown("Anomaly Score Line Chart")
        fig, ax = plt.subplots()
        ax.plot(df['Anomaly_Score'], color='red')
        ax.axhline(np.median(scores), linestyle='--', color='green', label='Median Threshold')
        ax.set_ylabel("Anomaly Score")
        ax.set_xlabel("Index")
        ax.legend()
        st.pyplot(fig)

        threshold = st.slider("Set Anomaly Threshold", float(np.min(scores)), float(np.max(scores)), float(np.median(scores)))
        flagged = df[df['Anomaly_Score'] < threshold]
        st.warning(f"{len(flagged)} anomalies detected below threshold.")

        if xgb_available:
            xgb_preds = clf.predict(df_scaled)
            df['XGB_Fault_Prediction'] = xgb_preds
            st.subheader("XGBoost Fault Classification")
            st.dataframe(df['XGB_Fault_Prediction'].value_counts().rename("Count").reset_index().rename(columns={"index": "Prediction"}))
        else:
            st.info("XGBoost model not available. Only anomaly detection is active.")

        st.subheader("Download Analysis Results")
        csv_out = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_out, file_name="isro_results.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Error during processing: {e}")
