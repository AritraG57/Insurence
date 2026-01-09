import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load("Linear_Regression_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Insurance Price Prediction")
st.markdown("Provide the following details to predict your insurance charges")

# ---------- USER INPUT ----------
age = st.slider("Age", 18, 100, 40)

gender = st.selectbox("Gender", ["Male", "Female"])
is_female = 1 if gender == "Female" else 0

bmi = st.slider("BMI", 10.0, 50.0, 24.0)

children = st.number_input("Number of Children", 0, 5, 0)

smoker = st.selectbox("Smoker", ["No", "Yes"])
is_smoker = 1 if smoker == "Yes" else 0

region = st.selectbox(
    "Region",
    ["northwest", "northeast", "southwest", "southeast"]
)

# ---------- INTERNAL LOGIC (NOT SHOWN IN UI) ----------

# Region encoding
region_southeast = 1 if region == "southeast" else 0

# BMI category encoding
bmi_category_Normal = 1 if 18 < bmi < 25 else 0
bmi_category_Overweight = 1 if 25 <= bmi < 30 else 0
bmi_category_Obese = 1 if bmi >= 30 else 0

# ---------- PREDICTION ----------
if st.button("Predict Insurance Charges"):

    input_data = {
        "age": age,
        "is_female": is_female,
        "bmi": bmi,
        "children": children,
        "is_smoker": is_smoker,
        "region_southeast": region_southeast,
        # "bmi_category_Normal": bmi_category_Normal,
        # "bmi_category_Overweight": bmi_category_Overweight,
        "bmi_category_Obese": bmi_category_Obese,
    }

    input_df = pd.DataFrame([input_data])

    # Ensure correct column order
    input_df = input_df[expected_columns]

    # Columns scaled during training
    scale_cols = ['age', 'bmi', 'children']

    # Copy input dataframe
    input_df_scaled = input_df.copy()

    # Scale ONLY numeric columns
    input_df_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

    # Ensure exact column order used in training
    input_df_scaled = input_df_scaled[
        ['age', 'is_female', 'bmi', 'children',
        'is_smoker', 'region_southeast', 'bmi_category_Obese']
    ]

    # Predict
    prediction = model.predict(input_df_scaled)[0]

    st.success(f"💰 Predicted Insurance Charges: ₹{prediction:.2f}")


    