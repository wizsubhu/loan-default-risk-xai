# loan-default-risk-xai
“ML model to predict loan defaults with SHAP explainability (86% AUC-ROC).”
# 🧠 Loan Default Risk Prediction with Explainable AI

A complete end-to-end machine learning project that predicts whether a loan applicant is likely to default, using data preprocessing, feature engineering, classification models (Logistic Regression, XGBoost), and model interpretability with SHAP. Visualized with interactive Plotly and deployed-ready components.

---

## 🚀 Project Highlights

- 🧹 Data Cleaning & Feature Engineering (Winsorization, DTI, Total Income)
- 📊 Exploratory Data Analysis (Radar Chart, Violin Plot, Correlation Heatmap, Sankey Diagram)
- ⚙️ Model Building: Logistic Regression, XGBoost with Class Imbalance Handling
- ✅ Evaluation: ROC Curve, AUC, Classification Report, Cross-Validation
- 🧠 Explainability: SHAP Summary, Dependence, and Force Plots
- 💡 Deployment-ready with Streamlit (optional enhancement)

---

## 📂 Dataset

- Source: Loan Prediction Dataset (binary classification)
- Target Variable: `Loan_Status` (Y/N → 1/0)
- Features: Demographics, Income, Loan Amount, Credit History, etc.

---

## 🛠️ Technologies & Tools

| Category           | Tools / Libraries                              |
|-------------------|--------------------------------------------------|
| Language          | Python                                           |
| Data Handling     | Pandas, NumPy, Scipy                             |
| Visualization     | Plotly, Matplotlib, Seaborn                      |
| ML Models         | Scikit-learn, XGBoost                            |
| Evaluation        | AUC, ROC, GridSearchCV, cross_val_score          |
| Explainability    | SHAP                                             |
| Deployment (Opt.) | Streamlit                                        |

---

## 📈 Visual Explorations

- **📊 Correlation Heatmap**
- **🎻 Violin Plot (DTI by Loan Status)**
- **🌐 Sankey Diagram** (Gender → Education flow)
- **📍 Radar Chart** (Financial indicators overview)

![Radar Chart](./assets/radar_chart.png)
![Violin Plot](./assets/violin_dti.png)

---

## 🤖 Machine Learning Pipeline

1. **Preprocessing**
   - Missing value imputation
   - Encoding categorical variables
   - Winsorization of income/loan features
   - Feature engineering (DTI, Total Income, Loan Term Years)

2. **Model Training**
   - Logistic Regression (baseline)
   - XGBoost (handling class imbalance via `scale_pos_weight`)
   - Hyperparameter tuning using GridSearchCV

3. **Evaluation**
   - ROC Curve, AUC, Classification Report
   - Cross-validation scores (AUC mean ± std)

---

## 🔍 Explainable AI with SHAP

SHAP was used to generate:

- **Global Feature Importance** (summary bar and dot plot)
- **Individual Explanation** (force plot for a test case)
- **Dependence Plot** (Credit History vs. SHAP values)

> 🧠 This helped uncover *why* the model predicted defaults, aiding trust and transparency.

![SHAP Summary](./assets/shap_global.png)
![Force Plot](./assets/shap_individual.png)

---

## 📦 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/loan-default-risk-shap.git
cd loan-default-risk-shap

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
python code4final.py
