import streamlit as st
import pandas as pd
import joblib
st.title("Academic Risk Prediction system ")

@st.cache_resource
def load_model():
    model= joblib.load("../models/academic_risk_model.pkl")
    return model

model=load_model()

st.header("📋 Student Academic Details")
col1,col2,col3=st.columns(3)

with col1:
    studytime = st.slider("📚 Weekly Study Time (1–4)", 1, 4, 2)

with col2:
    failures = st.slider("❌ Past Class Failures", 0, 4, 0)

with col3:
    absences = st.slider("🚫 Number of Absences", 0, 100, 5)

st.divider()



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

input_df=pd.DataFrame([input_data])
probs = model.predict_proba(input_df)[0]
risk_classes = model.classes_

confidence = dict(zip(risk_classes, probs))



def generate_explanation (studytime, failures, absences):
    reasons=[]

    if failures >= 2:
        reasons.append(" Multiple past academic failures")

    if absences >= 10:
        reasons.append(" High number of absences")

    if studytime <= 2:
        reasons.append(" Low weekly study time")

    if not reasons:
        if not reasons:
            if prediction == "High":
                return (
                    "The student is predicted as high risk due to a combination of "
                    "academic, demographic, and support-related factors learned by the model."
                )
            else:
                return "The student shows stable academic behavior with no major risk indicators."

    return "Prediction is influenced by :"+ ",".join(reasons)+"."

if st.button("🔍 Predict Academic Risk"):
    prediction = model.predict(input_df)[0]

    st.subheader("🎯 Prediction Result")

    if prediction == "High":
        st.error("⚠️ High Academic Risk")
    elif prediction == "Medium":
        st.warning("🟠 Medium Academic Risk")
    else:
        st.success("✅ Low Academic Risk")

    explanation = generate_explanation(
        studytime, failures, absences
    )

    st.subheader("📊 Prediction Confidence")

    confidence_df = pd.DataFrame({
        "Risk Level": list(confidence.keys()),
        "Probability (%)": [round(v * 100, 2) for v in confidence.values()]
    })

    st.table(confidence_df)
    st.bar_chart(confidence_df.set_index("Risk Level"))

    st.subheader("🧠 Explanation")
    st.info(explanation)


st.divider()
st.caption(
"⚠️ This system is a decision-support tool. "
    "Predictions are based on historical academic patterns "
    "and should be used alongside human judgement."
)



