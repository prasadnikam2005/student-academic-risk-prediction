import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# App Title
# -------------------------------
st.title("🎓 Academic Risk Prediction System")

# -------------------------------
# Load Model (Cloud-safe)
# -------------------------------
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "academic_risk_model.pkl"
    return joblib.load(model_path)

model = load_model()

# -------------------------------
# User Inputs
# -------------------------------
st.header("📋 Student Academic Details")

col1, col2, col3 = st.columns(3)

with col1:
    studytime = st.slider("📚 Weekly Study Time (1–4)", 1, 4, 2)

with col2:
    failures = st.slider("❌ Past Class Failures", 0, 4, 0)

with col3:
    absences = st.slider("🚫 Number of Absences", 0, 100, 5)

st.divider()

# -------------------------------
# Input Data (Model-Compatible)
# -------------------------------
input_data = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 2,
    "Fedu": 2,
    "Mjob": "services",
    "Fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": absences
}

input_df = pd.DataFrame([input_data])

# -------------------------------
# Explanation Logic (Aligned)
# -------------------------------
def generate_explanation(studytime, failures, absences, prediction):
    reasons = []

    if failures >= 2:
        reasons.append("multiple past academic failures")
    if absences >= 10:
        reasons.append("high number of absences")
    if studytime <= 2:
        reasons.append("low weekly study time")

    if reasons:
        return "Prediction influenced by: " + ", ".join(reasons) + "."
    else:
        return (
            f"The student is predicted as **{prediction} risk** based on "
            "overall academic, behavioral, and demographic patterns learned "
            "from historical data."
        )

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Academic Risk"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("🎯 Prediction Result")

    if prediction == "High":
        st.error("⚠️ High Academic Risk")
    elif prediction == "Medium":
        st.warning("🟠 Medium Academic Risk")
    else:
        st.success("✅ Low Academic Risk")

    # Confidence table
    confidence_df = pd.DataFrame({
        "Risk Level": model.classes_,
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    })

    st.subheader("📊 Prediction Confidence")
    st.table(confidence_df)
    st.bar_chart(confidence_df.set_index("Risk Level"))

    # Explanation
    st.subheader("🧠 Explanation")
    explanation = generate_explanation(studytime, failures, absences, prediction)
    st.info(explanation)

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption(
    "⚠️ This system is a decision-support tool. "
    "Predictions are based on historical academic patterns "
    "and should be used alongside human judgement."
)
