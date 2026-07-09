import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")

MODEL_DIR = "models"


@st.cache_resource
def load_artifacts():
    return {
        "clf_features": joblib.load(f"{MODEL_DIR}/classification_feature_columns.pkl"),
        "clf_scaler": joblib.load(f"{MODEL_DIR}/classification_scaler.pkl"),
        "param_encoder": joblib.load(f"{MODEL_DIR}/defining_param_encoder.pkl"),
        "meta": joblib.load(f"{MODEL_DIR}/deployment_metadata.pkl"),
        "reg_features": joblib.load(f"{MODEL_DIR}/linear_regression_feature_columns.pkl"),
        "reg_model": joblib.load(f"{MODEL_DIR}/linear_regression_model.pkl"),
        "reg_scaler": joblib.load(f"{MODEL_DIR}/linear_regression_scaler.pkl"),
        "target_encoder": joblib.load(f"{MODEL_DIR}/target_encoder.pkl"),
        "xgb_model": joblib.load(f"{MODEL_DIR}/xgb_model.pkl"),
    }


artifacts = load_artifacts()
meta = artifacts["meta"]

st.title("🌫️ Air Quality Index Predictor")
st.caption("XGBoost classification + Linear Regression, trained on US AQI data")

with st.form("aqi_form"):
    col1, col2 = st.columns(2)
    with col1:
        sites = st.number_input("Number of Sites Reporting", min_value=1, max_value=100, value=5)
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.05, format="%.4f")
        lng = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.24, format="%.4f")
        month = st.selectbox("Month", list(range(1, 13)), index=6)
    with col2:
        population = st.number_input("Population", min_value=0, value=100000, step=1000)
        density = st.number_input("Density (people/sq mi)", min_value=0.0, value=2000.0)
        pollutant = st.selectbox("Defining Parameter (pollutant)", list(artifacts["param_encoder"].classes_))

    submitted = st.form_submit_button("Predict AQI")

if submitted:
    def_encoded = artifacts["param_encoder"].transform([pollutant])[0]

    row = {
        "Number of Sites Reporting": sites,
        "lat": lat,
        "lng": lng,
        "population": population,
        "density": density,
        "month": month,
        "def_encoded": def_encoded,
    }

    # Classification (XGBoost)
    X_clf = pd.DataFrame([row])[artifacts["clf_features"]]
    X_clf_scaled = artifacts["clf_scaler"].transform(X_clf)
    class_pred = artifacts["xgb_model"].predict(X_clf_scaled)[0]
    class_label = artifacts["target_encoder"].inverse_transform([class_pred])[0]
    class_proba = artifacts["xgb_model"].predict_proba(X_clf_scaled)[0]

    # Regression (Linear Regression)
    X_reg = pd.DataFrame([row])[artifacts["reg_features"]]
    X_reg_scaled = artifacts["reg_scaler"].transform(X_reg)
    aqi_value = artifacts["reg_model"].predict(X_reg_scaled)[0]

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Predicted AQI Value", f"{aqi_value:.1f}")
    c2.metric("Predicted Category", class_label)

    with st.expander("Class probabilities"):
        proba_df = pd.DataFrame({
            "Category": artifacts["target_encoder"].inverse_transform(np.arange(len(class_proba))),
            "Probability": class_proba,
        }).sort_values("Probability", ascending=False)
        st.dataframe(proba_df, hide_index=True, use_container_width=True)

st.divider()
with st.expander("Model info"):
    st.write(f"XGBoost test accuracy: **{meta['xgb_test_accuracy']:.3f}**")
    st.write(f"Linear Regression MAE: **{meta['linear_regression_mae']:.2f}**")
    st.write(f"Linear Regression R²: **{meta['linear_regression_r2']:.3f}**")
