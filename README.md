# 🎓 Student Academic Risk Prediction System  
### Explainable • Cost-Sensitive • Deployment-Ready ML

> An end-to-end Machine Learning system that predicts **academic risk levels** using socio-academic and behavioral data — built with cross-validation rigor, cost-sensitive learning, and explainable outputs.

🔗 **Live App:**  
https://student-academic-risk-prediction-iheuopuzngcocm77g5bx4j.streamlit.app/

---

## 🚀 Why This Project Matters

Many students struggle due to hidden academic and behavioral patterns.  
Most institutions react **after failure occurs**.

This system enables:

- Early identification of high-risk students  
- Data-driven intervention decisions  
- Transparent, explainable predictions  
- Deployment-ready decision support  

The focus is not just accuracy — but **minimizing missed high-risk students**.

---

# 🎯 Problem Statement

Given socio-academic student data, classify students into:

- 🟢 **Low Risk**
- 🟠 **Medium Risk**
- 🔴 **High Risk**

### 🔴 Primary Objective:
Maximize **High-Risk Recall** to reduce false negatives.

In real-world intervention systems, **missing a high-risk student is costlier than raising a false alarm.**

---

# 📊 Dataset Overview

**Source:** UCI Machine Learning Repository – Student Performance Dataset  
**Total Records:** 395 students  

## 🎯 Target Engineering

- Risk categories derived from final grade (G3)
- Data leakage removed: `G1`, `G2`, `G3` excluded before modeling
- Multi-class classification setup

## 📌 Feature Categories

### Academic
- Study time
- Past failures
- Absences

### Behavioral
- Alcohol consumption
- Going out frequency
- Free time

### Socio-economic
- Parent education
- School support
- Internet access

---

# 🧠 Modeling Approach

## 🔹 Evaluation Strategy

- Stratified 5-Fold Cross-Validation  
- Results reported as Mean ± Standard Deviation  
- Class imbalance handled using balanced learning  

### 📈 Metrics Evaluated

- Accuracy  
- Macro F1  
- Balanced Accuracy  
- 🔴 High-Risk Recall (Primary Metric)

---

## 🔹 Model Comparison

| Model | High-Risk Recall | Macro F1 | Accuracy |
|-------|------------------|----------|----------|
| Decision Tree (controlled depth) | 0.40 | 0.38 | 0.47 |
| Random Forest (class-balanced) | 0.38 | 0.42 | 0.53 |

---

# ⚖️ Cost-Sensitive Decision Strategy

Default argmax prediction was replaced with a **threshold-based policy**.

### 📌 Decision Rule

Predict **High Risk** if:
P(High Risk) ≥ 0.35


This increases sensitivity toward high-risk students.

## 📊 After Threshold Calibration

| Metric | Value |
|--------|-------|
| 🔴 High-Risk Recall | **0.50** |
| Accuracy | 0.53 |
| Macro F1 | 0.44 |

This improves intervention sensitivity while maintaining model stability.

---

# 🔍 Explainability Layer

Designed for transparency and institutional usability.

Includes:

- Feature importance analysis  
- Rule-based explanation logic  
- Human-readable output summaries  
- Probability confidence visualization in Streamlit  

### 🔎 Top Risk Drivers

- Past academic failures  
- Absences  
- Low study time  
- High social activity patterns  

### 🧾 Example Output

> “Student predicted as High Risk due to multiple past failures and low weekly study time.”

---

# 🖥️ Deployment Architecture

- End-to-end sklearn Pipeline (preprocessing + classifier)
- Model serialized using `joblib`
- Deployed via Streamlit Cloud
- Threshold-based decision logic integrated inside app

## 🌐 App Features

- Interactive input sliders  
- Real-time risk prediction  
- Class probability visualization  
- Automated explanation generation  

---

# 🗂️ Project Structure
student-academic-risk-prediction/ ├── app/ │ └── app.py ├── models/ │ └── academic_risk_model.pkl ├── notebooks/ │ ├── 01_problem_understanding.ipynb │ ├── 02_eda.ipynb │ ├── 03_baseline_model.ipynb │ ├── 04_decision_tree.ipynb │ └── 05_knn_comparison.ipynb ├── requirements.txt └── README.md


---

# ⚠️ Limitations

- Small dataset (395 samples)
- Target derived from final grade (proxy outcome)
- No temporal progression modeling
- No external validation dataset

---

# 🔮 Future Enhancements

- Larger multi-institution dataset  
- Time-series modeling of academic progression  
- Probability calibration (Platt Scaling / Isotonic Regression)  
- SHAP-based explainability  
- Institutional analytics dashboard  

---

# 👨‍💻 Author

**Prasad**  
AIML Engineering Student  
Focused on Applied Machine Learning & Explainable AI  

---

⭐ If you found this project valuable, consider starring the repository.
