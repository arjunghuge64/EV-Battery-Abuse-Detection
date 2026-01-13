import streamlit as st
import numpy as np
import joblib
import pickle

# ===============================
# Load all models
# ===============================

# Battery Health Index & Charging Abuse
bhi_model = joblib.load("models/bhi/bhi_model.pkl")
abuse_model = joblib.load("models/bhi/abuse_model.pkl")

# Driving & Electrical Abuse
stress_score_pipeline = joblib.load("models/driving/stress_score_pipeline.pkl")
stress_class_pipeline = joblib.load("models/driving/stress_classification_pipeline.pkl")
label_encoder = joblib.load("models/driving/label_encoder.pkl")

# Charging Abuse Action Detection
charge_model = joblib.load("models/charging/charging_abuse_model123.pkl")
scaler = joblib.load("models/charging/scaler123.pkl")
features = pickle.load(open("models/charging/features.pkl", "rb"))

# ===============================
# Recommendation Engines
# ===============================

def bhi_recommendation(bhi):
    if bhi >= 85:
        return "Battery is healthy. Continue eco-friendly charging."
    elif bhi >= 70:
        return "Reduce fast charging and avoid overheating."
    elif bhi >= 50:
        return "Use slow charging and drive gently to protect battery."
    else:
        return "Battery is critical. Use AC charging only and visit service center."

def charging_recommendation(status):
    if status == "Normal":
        return "Charging behavior is safe. Keep using normal charging."
    else:
        return "Avoid fast charging above 80%, and do not charge at high temperature."

def driving_recommendation(label):
    if label == "Low":
        return "Driving behavior is safe. Keep driving smoothly."
    elif label == "Moderate":
        return "Avoid harsh acceleration and braking."
    else:
        return "Aggressive driving is damaging the battery. Drive gently."

# ===============================
# Streamlit Setup
# ===============================
st.set_page_config(page_title="EV Battery Intelligence", layout="wide")
st.title("🚗🔋 EV Battery Abuse & Health Intelligence System")

menu = st.sidebar.radio("Select Module", [
    "Battery Health Index",
    "Charging Abuse Actions",
    "Driving & Electrical Abuse",
    "About Project"
])

# ===============================
# Module 1: Battery Health Index
# ===============================
if menu == "Battery Health Index":
    st.header("🔋 Battery Health Index (BHI)")

    temp = st.slider("Battery Temperature (°C)", 10, 50, 30)
    cycles = st.slider("Charging Cycles", 0, 3000, 500)
    fast = st.slider("Fast Charge Ratio", 0.0, 1.0, 0.3)
    discharge = st.slider("Discharge Rate (C)", 0.5, 3.0, 1.2)
    age = st.slider("Vehicle Age (Months)", 0, 120, 24)
    resistance = st.slider("Internal Resistance (Ohm)", 0.01, 0.1, 0.03)
    driving = st.radio("Driving Style", ["Conservative","Moderate","Aggressive"])
    driving_map = {"Conservative":0.2,"Moderate":0.5,"Aggressive":1.0}

    if st.button("Predict Battery Health"):
        X = np.array([[temp, cycles, fast, discharge, age, resistance, driving_map[driving]]])
        bhi = bhi_model.predict(X)[0]
        abuse = abuse_model.predict(X[:,2:7])[0]
        bhi_adj = bhi * (1 - 0.25 * abuse)

        st.metric("Battery Health Index", f"{bhi_adj:.2f}")
        st.progress(min(int(bhi_adj),100))

        if bhi_adj >= 85:
            st.success("🟢 Healthy Battery")
        elif bhi_adj >= 70:
            st.warning("🟡 Moderate Battery")
        else:
            st.error("🔴 Battery Degrading")

        st.info("🌱 Recommendation: " + bhi_recommendation(bhi_adj))

# ===============================
# Module 2: Charging Abuse Actions
# ===============================
elif menu == "Charging Abuse Actions":
    st.header("⚡ Charging Abuse Detection & Eco Recommendations")

    user_inputs = []
    for feature in features:
        user_inputs.append(st.number_input(feature, value=0.5))

    if st.button("Analyze Charging"):
        data = np.array(user_inputs).reshape(1,-1)
        data_scaled = scaler.transform(data)
        pred = charge_model.predict(data_scaled)[0]

        st.metric("Charging Status", pred)

        if pred == "Abusive":
            st.error("⚠ Abusive Charging Detected")
        else:
            st.success("Charging is Safe")

        st.info("🌱 Recommendation: " + charging_recommendation(pred))

# ===============================
# Module 3: Driving Abuse
# ===============================
elif menu == "Driving & Electrical Abuse":
    st.header("🚗 Driving & Electrical Stress Detection")

    current = st.number_input("Average Current")
    peak = st.number_input("Peak Current")
    min_v = st.number_input("Minimum Voltage")
    soc_drop = st.number_input("SOC Drop")
    ac = st.number_input("AC Usage")

    if st.button("Analyze Driving"):
        data = np.array([[current, peak, min_v, soc_drop, ac]])
        stress = stress_score_pipeline.predict(data)[0]
        cls = stress_class_pipeline.predict(data)[0]
        label = label_encoder.inverse_transform([cls])[0]

        st.metric("Stress Score", f"{stress:.2f}")
        st.metric("Abuse Level", label)
        st.info("🌱 Recommendation: " + driving_recommendation(label))

# ===============================
# About Project
# ===============================
else:
    st.header("📘 About This Project")
    st.write("""
    This EV Intelligence System uses Machine Learning and Green AI to:
    - Detect charging and driving abuse
    - Predict Battery Health Index (BHI)
    - Provide eco-friendly recommendations

    It integrates three intelligent models into a single web platform
    to help users protect EV batteries and promote sustainable EV usage.
    """)
