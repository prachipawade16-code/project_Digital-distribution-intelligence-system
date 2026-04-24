import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Dark Mode Toggle (NEW 🔥)
# -------------------------------
dark_mode = st.checkbox("🌙 Enable Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3, label {
            color: white !important;
        }
        .stButton>button {
            background-color: #222;
            color: white;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #eef2f3, #dfe9f3);
        }
        h1, h2, h3 {
            text-align: center;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Load Model & Scaler
# -------------------------------
@st.cache_resource
def load_files():
    with open("notebook/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("notebook/model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model


# -------------------------------
# Rule-Based Logic
# -------------------------------
def rule_based_label(screen, social, notif, switch, sleep, work):
    score = (
        screen * 0.4 +
        social * 0.3 +
        notif * 0.01 +
        switch * 0.01 -
        sleep * 0.3 -
        work * 0.1
    )

    if score < 5:
        return 0
    elif score < 8:
        return 1
    else:
        return 2


# -------------------------------
# Prediction Function
# -------------------------------
def predict_distraction(data, scaler, model):
    x_scaled = scaler.transform(data)
    pred = model.predict(x_scaled)[0]
    probs = model.predict_proba(x_scaled)[0]
    confidence = probs[pred]
    return pred, confidence, probs


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Digital Distraction", layout="centered")

st.title("📱 Digital Distraction Intelligence System")

# Header Image
st.image("https://cdn-icons-png.flaticon.com/512/3209/3209265.png", use_container_width=True)

st.markdown("### 🧠 Analyze your digital habits")

# Input Image
st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=100)

# Inputs
age = st.number_input("Age", 16, 40, 22)
screen = st.slider("Daily Screen Time (hrs)", 1, 12, 6)
social = st.slider("Social Media Time (hrs)", 1, 8, 3)
notif = st.slider("Notifications per Day", 10, 200, 80)
switch = st.slider("App Switches", 5, 120, 40)
sleep = st.slider("Sleep Hours", 3, 10, 7)
work = st.slider("Work Hours", 1, 12, 6)

# -------------------------------
# Predict
# -------------------------------
if st.button("🔍 Predict"):
    scaler, model = load_files()

    input_df = pd.DataFrame({
        'Age': [age],
        'Daily_Screen_Time': [screen],
        'Social_Media_Time': [social],
        'Notifications': [notif],
        'App_Switches': [switch],
        'Sleep_Hours': [sleep],
        'Work_Hours': [work]
    })

    pred, confidence, probs = predict_distraction(input_df, scaler, model)

    rule_pred = rule_based_label(screen, social, notif, switch, sleep, work)

    if confidence < 0.6:
        final_pred = rule_pred
        st.info("Using rule-based prediction (model low confidence)")
    else:
        final_pred = pred

    # Result Image
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)

    if final_pred == 1:
        st.error("⚠️ Highly Distracted")
        st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=120)
    else:
        st.success("✅ Not Distracted")
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=120)

    st.subheader(f"Confidence: {confidence:.2f}")
    st.progress(float(confidence))

    # Pie Chart
    st.subheader("📊 Prediction Probability")
    labels = ["Not Distracted", "Highly Distracted"]
    sizes = probs if len(probs) == 2 else [1 - confidence, confidence]

    fig3, ax3 = plt.subplots()
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig3)


# -------------------------------
# Visualization
# -------------------------------
st.subheader("📊 Data Visualization")

st.image("https://cdn-icons-png.flaticon.com/512/2721/2721297.png", width=100)

try:
    df = pd.read_csv("Notebook/Digital_Distraction.csv")

    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Daily_Screen_Time"], df["Distraction_Level"])
    ax1.set_title("Screen Time vs Distraction")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(df["Notifications"])
    ax2.set_title("Notifications Distribution")
    st.pyplot(fig2)

    counts = df["Distraction_Level"].value_counts()

    fig4, ax4 = plt.subplots()
    ax4.pie(counts, labels=["Not Distracted", "Highly Distracted"], autopct='%1.1f%%')
    st.pyplot(fig4)

    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=100)

except:
    st.warning("Dataset not found.")