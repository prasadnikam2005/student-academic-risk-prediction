import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Academic Risk Prediction", layout="centered")
st.title("Academic Risk Prediction System")

THRESHOLD_HIGH = 0.35  # tuned threshold to improve High-risk recall

@st.cache_resource
def load_model():
    # Robust path: app/streamlit_app.py -> ../models/academic_risk_model.pkl
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "models", "academic_risk_model.pkl")
    model = joblib.load(model_path)
    return model

model = load_model()

st.header("📋 Student Academic Details")
col1, col2, col3 = st.columns(3)

with col1:
    studytime = st.slider("📚 Weekly Study Time (1–4)", 1, 4, 2)

with col2:
    failures = st.slider("❌ Past Class Failures", 0, 4, 0)

with col3:
    absences = st.slider("🚫 Number of Absences", 0, 100, 5)

st.caption(f"Decision policy: predict **High** if P(High) ≥ {THRESHOLD_HIGH:.2f} (cost-sensitive threshold)")

st.divider()

# Default values for all other features (keeps UI minimal but model input complete)
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

def get_classes_from_model(m):
    """
    Works for sklearn Pipeline where final estimator is named 'classifier'
    and also for direct estimators that expose classes_.
    """
    if hasattr(m, "named_steps") and "classifier" in m.named_steps:
        return m.named_steps["classifier"].classes_
    if hasattr(m, "classes_"):
        return m.classes_
    # fallback: try last step in pipeline
    if hasattr(m, "steps") and len(m.steps) > 0:
        last_est = m.steps[-1][1]
        if hasattr(last_est, "classes_"):
            return last_est.classes_
    raise AttributeError("Could not find classes_ on model. Check pipeline step name.")

def generate_explanation(studytime_val, failures_val, absences_val, prediction_val):
    reasons = []

    if failures_val >= 2:
        reasons.append("multiple past academic failures")
    if absences_val >= 10:
        reasons.append("high number of absences")
    if studytime_val <= 2:
        reasons.append("low weekly study time")

    if not reasons:
        if prediction_val == "High":
            return ("High risk predicted based on patterns across academic history, attendance, "
                    "and support-related factors learned by the model.")
        return "No strong risk indicators detected from study time, failures, and absences."

    return "Prediction is influenced by: " + ", ".join(reasons) + "."

if st.button("🔍 Predict Academic Risk"):
    # Predict probabilities
    probs = model.predict_proba(input_df)[0]
    risk_classes = get_classes_from_model(model)

    # Threshold-based decision for High class
    high_idx = list(risk_classes).index("High")
    high_prob = float(probs[high_idx])

    if high_prob >= THRESHOLD_HIGH:
        prediction = "High"
    else:
        prediction = str(risk_classes[int(probs.argmax())])

    st.subheader("🎯 Prediction Result")
    if prediction == "High":
        st.error("⚠️ High Academic Risk")
    elif prediction == "Medium":
        st.warning("🟠 Medium Academic Risk")
    else:
        st.success("✅ Low Academic Risk")

    # Confidence table + chart
    confidence = dict(zip(risk_classes, probs))
    confidence_df = pd.DataFrame({
        "Risk Level": list(confidence.keys()),
        "Probability (%)": [round(float(v) * 100, 2) for v in confidence.values()]
    })

    st.subheader("📊 Prediction Confidence")
    st.table(confidence_df)
    st.bar_chart(confidence_df.set_index("Risk Level"))

    # Explanation
    explanation = generate_explanation(studytime, failures, absences, prediction)
    st.subheader("🧠 Explanation")
    st.info(explanation)

st.divider()
st.caption(
    "⚠️ This system is a decision-support tool. Predictions are based on historical academic patterns "
    "and should be used alongside human judgement."
)