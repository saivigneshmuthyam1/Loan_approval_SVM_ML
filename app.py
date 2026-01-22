import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. APP TITLE & DESCRIPTION
# -------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("🏦 Smart Loan Approval System")
st.write("""
This system uses **Support Vector Machines (SVM)** to predict whether a loan 
will be approved based on applicant details.
""")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("train.csv")

# DROP ID COLUMN (VERY IMPORTANT)
df.drop("Loan_ID", axis=1, inplace=True)

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df_encoded.drop("Loan_Status_Y", axis=1)
y = df_encoded["Loan_Status_Y"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 2. INPUT SECTION (SIDEBAR)
# -------------------------------
st.sidebar.header("Enter Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Employed", "Self Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# 3. MODEL SELECTION
# -------------------------------
kernel_choice = st.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

kernel_map = {
    "Linear SVM": "linear",
    "Polynomial SVM": "poly",
    "RBF SVM": "rbf"
}

# -------------------------------
# Prepare Input Vector
# -------------------------------
def prepare_input():
    input_data = {
        'ApplicantIncome': income,
        'CoapplicantIncome': 0,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': 360,
        'Credit_History': 1 if credit_history == "Yes" else 0,
        'Gender_Male': 1,
        'Married_Yes': 1,
        'Dependents_1': 0,
        'Dependents_2': 0,
        'Dependents_3+': 0,
        'Education_Not Graduate': 0,
        'Self_Employed_Yes': 1 if employment == "Self Employed" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]  # align columns perfectly
    input_scaled = scaler.transform(input_df)
    return input_scaled

# -------------------------------
# 4. PREDICTION BUTTON
# -------------------------------
if st.button("🔍 Check Loan Eligibility"):

    model = SVC(kernel=kernel_map[kernel_choice], probability=True)
    model.fit(X_train, y_train)

    user_input = prepare_input()
    prediction = model.predict(user_input)[0]
    confidence = model.predict_proba(user_input)[0].max()

    # -------------------------------
    # 5. OUTPUT SECTION
    # -------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Confidence Score:** {confidence:.2f}")
    st.write(f"**Kernel Used:** {kernel_choice}")

    # -------------------------------
    # 6. BUSINESS EXPLANATION
    # -------------------------------
    st.subheader("Business Explanation")

    if prediction == 1:
        st.info("""
        Based on the applicant's **income level and positive credit history**, 
        the system predicts that the applicant is **likely to repay the loan**.
        """)
    else:
        st.warning("""
        Due to **low income and/or negative credit history**, 
        the system predicts that the applicant is **unlikely to repay the loan**.
        """)
