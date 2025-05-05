
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import shap 
from sklearn.model_selection import train_test_split
from scipy.stats import mstats  
from sklearn.metrics import RocCurveDisplay, classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb


# Load the dataset from Google Drive or upload directly to Colab
from google.colab import files
uploaded = files.upload()  # Upload 'train_u6lujuX_CVtuZ9i.csv'

df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')


# Drop Loan_ID (it's just an identifier)
df.drop('Loan_ID', axis=1, inplace=True)

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Encode Categorical Variables 
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

#  Handle Outliers 
# Winsorize numerical features (cap extreme values at 5th/95th percentiles)
df['LoanAmount'] = mstats.winsorize(df['LoanAmount'], limits=[0.05, 0.05])
df['ApplicantIncome'] = mstats.winsorize(df['ApplicantIncome'], limits=[0.05, 0.05])
df['CoapplicantIncome'] = mstats.winsorize(df['CoapplicantIncome'], limits=[0.05, 0.05])

# Convert Loan_Status to binary values
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Feature Engineering
# Create TotalIncome
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Handle zero or negative TotalIncome to avoid division errors in DTI calculation
df = df[df['TotalIncome'] > 0]

# Create Debt-to-Income Ratio (DTI)
df['DTI'] = df['LoanAmount'] / df['TotalIncome']

# Create Loan Term in Years
df['LoanTermYears'] = df['Loan_Amount_Term'] / 12

# Correlation Heatmap
corr_matrix = df.corr(numeric_only=True)
fig3 = px.imshow(corr_matrix, text_auto=True, 
                title='Feature Correlation Heatmap')
fig3.show()

#DTI Distribution by Loan Status
fig5 = px.violin(df, x='Loan_Status', y='DTI', 
                box=True, title='Debt-to-Income Ratio by Loan Status')
fig5.show()

# Prepare data for Sankey Diagram for categorical data
# Use the one-hot encoded columns for Gender and Education
sankey_data = df.groupby(['Gender_Male', 'Gender_Female', 
                           'Education_Graduate', 'Education_Not Graduate']).size().reset_index(name='Count')

# Define nodes for Sankey diagram
gender_labels = ['Male', 'Female']  # Use original labels
education_labels = ['Graduate', 'Not Graduate'] # Use original labels
all_labels = gender_labels + education_labels

#Define node and link for Sankey diagram
node = dict(
    pad=15,
    thickness=20,
    line=dict(color="black", width=0.5),
    label=all_labels,
    color=["#636EFA", "#EF553B"]  # Colors for Gender nodes
)

# Use 'Gender_Male' and 'Gender_Female' to determine the source index
source_indices = sankey_data.apply(lambda row: gender_labels.index('Male') if row['Gender_Male'] == 1 else gender_labels.index('Female'), axis=1)  
target_indices = sankey_data.apply(lambda row: education_labels.index('Graduate') if row['Education_Graduate'] == 1 else education_labels.index('Not Graduate'), axis=1) + len(gender_labels)

link = dict(
    source=source_indices,
    target=target_indices,
    value=sankey_data['Count'],
    color=["rgba(99,110,250,0.8)" if row['Gender_Male'] == 1 else "rgba(239,85,59,0.8)" for index, row in sankey_data.iterrows()]  # Use iterrows for color assignment
)

# Create Sankey diagram
sankey_fig = go.Figure(go.Sankey(node=node, link=link))
sankey_fig.update_layout(title_text="Sankey Diagram: Gender to Education Flow", font_size=10)

# Prepare data for Radar Chart for numerical data
# Select relevant columns and drop rows with missing values
radar_data = df[['Credit_History', 'DTI', 'LoanAmount', 'TotalIncome', 'LoanTermYears']].dropna()

# Normalize data for Radar chart
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

radar_data_normalized = radar_data.apply(normalize)

# Melt the DataFrame for Plotly
radar_melted = radar_data_normalized.melt(var_name='Attribute', value_name='Value')

# Create Radar chart
radar_fig = px.line_polar(radar_melted, r='Value', theta='Attribute',  render_mode='svg')
radar_fig.update_traces(fill='toself')
radar_fig.update_layout(title='Radar Chart: Comparison of Financial Indicators')

# Train-Test Split 
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Model 1: Logistic Regression 
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print(f"AUC-ROC: {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.2f}")

# Model 2: XGBoost 
# Handle class imbalance
scale_pos_weight = sum(y == 0) / sum(y == 1)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"AUC-ROC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]):.2f}")

#  ROC Curve
RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)
plt.title(f'XGBoost ROC Curve (AUC = {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]):.2f})')
plt.show()

from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.8]
}

grid = GridSearchCV(xgb_model, params, scoring='roc_auc', cv=5)
grid.fit(X_train, y_train)
print(f"Best AUC-ROC: {grid.best_score_:.2f}")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
print(f"Cross-Validated AUC-ROC: {np.mean(scores):.2f} (±{np.std(scores):.2f})")

#  SHAP Explainability 

def explain_model(model, X_train, X_test, model_type='xgb'):
    """
    Generates SHAP explanations for global and local interpretability.
    """
    # Initialize explainer based on model type
    if model_type == 'xgb':
        explainer = shap.TreeExplainer(model)
    else:  # For non-tree models (e.g., logistic regression)
        explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(50))

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # 1. Global Feature Importance (Summary Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP Values)", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_global.png', dpi=300)
    plt.show()

    # 2. Detailed Summary Plot (Dot Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Feature Impact on Loan Decisions", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_detailed.png', dpi=300)
    plt.show()

    # 3. Individual Explanation (Force Plot)
    sample_idx = 10  # Example: Explain 10th test case
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[sample_idx], 
        X_test.iloc[sample_idx], 
        matplotlib=True, 
        show=False
    )
    plt.title(f"Loan Decision Explanation for Applicant #{sample_idx}", fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_individual.png', dpi=300)
    plt.show()

    # 4. Dependence Plot for Key Feature
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        "Credit_History",  # Most important feature
        shap_values, 
        X_test, 
        interaction_index='auto',
        show=False
    )
    plt.title("Credit History Impact with Auto Interaction", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_dependence.png', dpi=300)
    plt.show()

    return explainer, shap_values

# Generate explanations for XGBoost
xgb_explainer, xgb_shap_values = explain_model(xgb_model, X_train, X_test)

