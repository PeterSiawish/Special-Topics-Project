# gui_app.py

import streamlit as st
import joblib
import numpy as np

# Load trained KNN model
model = joblib.load("personality_model.pkl")

# Page configuration
st.set_page_config(page_title="Personality Predictor", layout="centered")

st.title("🧠 Personality Predictor")
st.write("Enter your details to find out if you're an Introvert or Extrovert.")

# Input form
with st.form("personality_form"):
    time_spent_alone = st.slider("How many hours do you spend alone daily?", 0, 11, 5)
    stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
    social_event_attendance = st.slider("How often do you attend social events? (0-10)", 0, 10, 5)
    going_outside = st.slider("How many days per week do you go outside? (0-7)", 0, 7, 3)
    drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
    friends_circle_size = st.slider("How many close friends do you have? (0-15)", 0, 15, 5)
    post_frequency = st.slider("How often do you post on social media? (0-10)", 0, 10, 5)

    submitted = st.form_submit_button("Predict Personality")

# Prediction logic
if submitted:
    # Convert categorical to numeric
    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained_after_socializing == "Yes" else 0

    # Arrange features in the correct order
    features = np.array([[time_spent_alone, stage_fear_val, social_event_attendance,
                          going_outside, drained_val, friends_circle_size, post_frequency]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]  # Get prediction probabilities
    confidence = np.max(proba) * 100  # Highest probability as confidence

    # Label mapping
    label_map = {0: "You are an Introvert 🤔", 1: "You are an Extrovert 😄"}

    # Display results
    st.success(f"🧬 Prediction: {label_map[prediction]}")
    st.info(f"🔍 Model Confidence: **{confidence:.2f}%**")
 