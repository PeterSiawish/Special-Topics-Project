# 🧠 Personality Predictor App

Welcome! This is a simple Streamlit web application designed to predict whether a user leans more towards **Introversion** or **Extroversion** based on a series of behavioral and lifestyle questions.

_**Disclaimer:** This model is for illustrative purposes only and should not be used as a substitute for professional psychological assessment._

---

## ✨ Features

- **Interactive Interface:** A user-friendly graphical interface built with Streamlit.
- **Personality Classification:** Predicts "Introvert" or "Extrovert" using a pre-trained machine learning model.
- **Model Confidence:** Displays the prediction's confidence score to the user.
- **Personalized Suggestions:** Offers brief, relevant advice based on the predicted personality type.
- **Clear & Intuitive Questions:** Easy-to-understand input fields with helpful tooltips.

---

## 🚀 How It Works

The application gathers user input through various sliders and selection boxes (e.g., daily alone time, social event attendance, stage fear). These inputs are then processed and fed into a pre-trained **K-Nearest Neighbors (KNN)** machine learning model, which is loaded from a `personality_model.pkl` file. The model outputs a personality classification along with a confidence score, and the app presents this information along with tailored suggestions.

---

## ⚙️ Technologies Used

- **Python**
- **Streamlit** (for the web application GUI)
- **scikit-learn** (for the machine learning model)
- **joblib** (for model serialization/deserialization)
- **NumPy** (for numerical operations)

---

## 💻 Getting Started

Follow these steps to get a local copy of the project up and running on your machine.

### Prerequisites

Make sure you have **Python 3.8 or newer** installed on your system.

### 1. Clone the Repository

First, clone this GitHub repository to your local machine using Git:

```bash
git clone https://github.com/PeterSiawish/Special-Topics-Project
cd Special-Topics-Project
```

### 2. Create a Virtual Environment (Recommended)

It's best practice to create a dedicated virtual environment for your project to manage dependencies:

```bash
python -m venv venv
```

Then activate the virtual environment:
On Windows:

```bash
.\venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

With your virtual environment activated, install all the necessary Python libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

Once all dependencies are installed, you can launch the Streamlit application:

```bash
streamlit run gui_app.py
```

This command will automatically open the application in your default web browser, usually at http://localhost:8501.
