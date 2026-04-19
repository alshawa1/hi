import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Starting to train the ML Model for deployment...")

# 1. Load Data
df = pd.read_csv('bank-additional-full (1).csv', sep=';')

# 2. Advanced Preprocessing
if 'duration' in df.columns:
    df = df.drop('duration', axis=1)

df['target'] = df['y'].map({'yes': 1, 'no': 0})
df = df.drop('y', axis=1)

# Impute unknowns
for col in ['job', 'marital', 'housing', 'loan']:
    mode_val = df[df[col] != 'unknown'][col].mode()[0]
    df[col] = df[col].replace('unknown', mode_val)

# Handle pdays
df['contacted_before'] = (df['pdays'] != 999).astype(int)
df.loc[df['pdays'] == 999, 'pdays'] = 0

# Ordinal Education
edu_map = {'illiterate':0, 'unknown':1, 'basic.4y':2, 'basic.6y':3, 'basic.9y':4, 'high.school':5, 'professional.course':6, 'university.degree':7}
df['education_level'] = df['education'].map(edu_map)
df = df.drop('education', axis=1)

# Cap outliers
df.loc[df['campaign'] > 15, 'campaign'] = 15

# Drop multicollinear econ features
df = df.drop(['emp.var.rate', 'nr.employed'], axis=1)

# 3. Model Prep
df_ml = df.drop(columns=['age_group'], errors='ignore')

categorical_cols = df_ml.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

expected_columns = list(X.columns)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train best model
print("Training Logistic Regression Model...")
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
model.fit(X_scaled, y)

# 5. Save Artifacts for Streamlit
joblib.dump(model, 'log_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(expected_columns, 'expected_columns.pkl')

print("✅ Model, Scaler, and Columns saved successfully!")
print("Run 'streamlit run app.py' to start the web app.")
