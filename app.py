import streamlit as st
import joblib
import os

st.title("📊 Customer Churn Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "churn_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_names.pkl"
)

st.write("App location:")
st.code(BASE_DIR)

st.write("Model location:")
st.code(MODEL_PATH)

# Check model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file NOT found")
    st.stop()

if not os.path.exists(FEATURE_PATH):
    st.error("❌ Feature file NOT found")
    st.stop()

# Load files
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

st.success("✅ Model loaded successfully!")

st.write(
    "Number of features:",
    len(feature_names)
)

st.write(
    "Model type:",
    type(model).__name__
)
