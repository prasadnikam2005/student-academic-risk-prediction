# 🎓 Student Academic Risk Prediction System (Explainable ML)

An end-to-end, explainable Machine Learning system that predicts **academic risk levels** of students using historical academic and behavioral data.  
The system is designed as a **decision-support tool** for early academic intervention and counseling.

🔗 **Live App**:  
https://student-academic-risk-prediction-iheuopuzngcocm77g5bx4j.streamlit.app/

---

## 🚀 Project Motivation

Many students underperform or drop out due to unnoticed academic and behavioral patterns.  
Early identification of **high-risk students** allows institutions and counselors to intervene proactively.

This project focuses on:
- Early risk prediction
- Explainable ML (not black-box)
- Real-world deployment

---

## 🎯 Problem Statement

Given a student's academic, behavioral, and socio-demographic data,  
**predict the student's academic risk level** as:

- 🟢 Low Risk  
- 🟠 Medium Risk  
- 🔴 High Risk  

and **explain why** the student is predicted to be at risk.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository – Student Performance Dataset
- **Type**: Real-world educational data
- **Records**: 395 students
- **Features**:
  - Academic: study time, failures, absences, grades
  - Behavioral: alcohol consumption, free time, outings
  - Family & Socio-economic: parental education, family support, internet access

> The dataset is widely used in academic research and is interview-safe.

---

## 🧠 Machine Learning Approach

### 🔹 Target Variable
Final academic performance was converted into **risk categories**:
- High Risk
- Medium Risk
- Low Risk

### 🔹 Models Implemented
| Model | Purpose |
|------|--------|
| Logistic Regression | Baseline model |
| Decision Tree | **Primary model (explainable)** |
| KNN | Comparison model |

### 🔹 Why Decision Tree?
- Handles non-linear relationships
- Interpretable feature importance
- Easy to explain in interviews
- Suitable for decision-support systems

---

## 📈 Model Evaluation

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

📌 Emphasis was placed on **recall for high-risk students**, as missing a high-risk student is costlier than a false alarm.

---

## 🔍 Explainability (Key Differentiator)

The system provides:
- Feature importance from Decision Tree
- Rule-based explanations (failures, absences, study time)
- Prediction confidence (class probabilities)

Example:
> “Student predicted as high risk due to multiple past failures and high number of absences.”

---

## 🖥️ Deployment

- **Framework**: Streamlit
- **Hosting**: Streamlit Cloud (Free)
- **Deployment Type**: Interactive decision-support web app

### App Capabilities:
- User input via sliders
- Risk prediction
- Confidence visualization
- Human-readable explanations

🔗 **Live App**:  
https://student-academic-risk-prediction-iheuopuzngcocm77g5bx4j.streamlit.app/

---

## 🗂️ Project Structure
student-academic-risk-prediction/
├── app/
│ └── app.py
├── models/
│ └── academic_risk_model.pkl
├── notebooks/
│ ├── 01_problem_understanding.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_baseline_model.ipynb
│ ├── 04_decision_tree.ipynb
│ └── 05_knn_comparison.ipynb
├── requirements.txt
└── README.md


---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- GitHub

---

## ⚠️ Ethical Considerations

- Predictions should **not be used for punishment**
- The system is a **decision-support tool**, not a final authority
- Human judgment is essential
- Risk of bias due to limited dataset size is acknowledged

---

## 🔮 Future Improvements

- Larger, multi-institution dataset
- Time-series performance tracking
- College-level dashboards
- SHAP-based explainability
- Integration with counseling platforms

---

## 👤 Author

**Prasad**  
AIML Engineering Student  
Interested in Explainable AI, Applied ML, and Real-world Decision Systems

---

## ⭐ If you find this project useful
Give it a ⭐ on GitHub and feel free to explore or extend it!
