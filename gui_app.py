import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Personality Predictor", layout="centered")

# Load trained KNN model
# Use st.cache_resource to cache the model, preventing it from reloading on every interaction
@st.cache_resource
def load_personality_model():
    with st.spinner("Loading personality model..."):
        return joblib.load("personality_model.pkl")

model = load_personality_model()

st.title("🧠 Personality Predictor")

st.markdown("""
Welcome! This simple model attempts to predict whether you lean more towards **Introversion** or **Extroversion** based on a few lifestyle questions. 
Please answer honestly to get the most accurate prediction.

_This model is for illustrative purposes only and should not be used as a substitute for professional psychological assessment._
""")

st.divider() # Add a horizontal divider for better visual separation

# Input form
with st.form("personality_form"):
    st.header("Tell Us About Yourself") # Added a sub-header for the form

    time_spent_alone = st.slider(
        "⏳ How many hours do you typically spend alone daily?", 
        0, 11, 5, help="Consider time spent in solitude, away from others."
    )
    stage_fear = st.selectbox(
        "🎤 Do you experience stage fear or significant anxiety in front of a group?", 
        ["Yes", "No"]
    )
    social_event_attendance = st.slider(
        "🎉 How often do you attend social events (parties, gatherings, meetings, etc.)? (0 = Never, 10 = Very Often)", 
        0, 10, 5, help="Rate your frequency on a scale from 0 to 10."
    )
    going_outside = st.slider(
        "🚶‍♀️ How many days per week do you typically go outside your home for non-work/school activities?", 
        0, 7, 3, help="Think about leisure activities, walks, or meeting friends."
    )
    drained_after_socializing = st.selectbox(
        "🔋 Do you feel emotionally drained or tired after extensive socializing?", 
        ["Yes", "No"]
    )
    friends_circle_size = st.slider(
        "👫 How many close friends would you say you have?", 
        0, 15, 5, help="These are friends you feel truly close to and confide in."
    )
    post_frequency = st.slider(
        "📱 How often do you post on social media (e.g., Facebook, Instagram, X) per week?", 
        0, 10, 5, help="Estimate your average posting frequency."
    )

    st.markdown("---") # Another divider before the submit button
    submitted = st.form_submit_button("🔮 Predict My Personality")

# Prediction logic
if submitted:
    # Convert categorical to numeric
    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained_after_socializing == "Yes" else 0

    # Arrange features in the correct order
    features = np.array([[time_spent_alone, stage_fear_val, social_event_attendance,
                           going_outside, drained_val, friends_circle_size, post_frequency]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = np.max(proba) * 100

    # Label mapping and description
    label_map = {
        0: "You are likely an **Introvert** 🤔", 
        1: "You are likely an **Extrovert** 😄"
    }
    
    personality_description = {
        0: "Introverts often recharge by spending time alone and may prefer deeper, more intimate social interactions over large gatherings.",
        1: "Extroverts typically gain energy from social interaction and tend to enjoy being around people and in lively environments."
    }

    st.success(f"### {label_map[prediction]}") # Larger, bolder prediction
    st.info(f"🔍 Model Confidence: **{confidence:.2f}%**")
    st.markdown(f"*{personality_description[prediction]}*") # Add a brief description
    
    st.balloons() # Fun little animation for success!