# API Reference & Feature Specification

## Input Parameter Specifications

### Age
- **Type**: Integer
- **Range**: 18 to 100 years
- **Default**: 40
- **Widget**: Slider
- **Requirement**: Required
- **Preprocessing**: Scaled using StandardScaler (training mean/std)

### Gender
- **Type**: Categorical
- **Options**: "Male", "Female"
- **Default**: "Male"
- **Widget**: Select Box
- **Requirement**: Required
- **Encoding**:
  - Male → `is_female = 0`
  - Female → `is_female = 1`

### BMI (Body Mass Index)
- **Type**: Float
- **Range**: 10.0 to 50.0
- **Default**: 24.0
- **Widget**: Slider
- **Requirement**: Required
- **Preprocessing**: Scaled using StandardScaler
- **Derived Feature**: bmi_category_Obese
  - True if BMI ≥ 30, False otherwise

### Number of Children
- **Type**: Integer
- **Range**: 0 to 5
- **Default**: 0
- **Widget**: Number Input
- **Requirement**: Required
- **Preprocessing**: Scaled using StandardScaler (training mean/std)

### Smoker Status
- **Type**: Categorical
- **Options**: "No", "Yes"
- **Widget**: Select Box
- **Requirement**: Required
- **Encoding**:
  - No → `is_smoker = 0`
  - Yes → `is_smoker = 1`

### Region
- **Type**: Categorical
- **Options**: "northwest", "northeast", "southwest", "southeast"
- **Widget**: Select Box
- **Requirement**: Required
- **Encoding**: One-hot encoding (region_southeast only used)
  - Southeast → `region_southeast = 1`
  - Other → `region_southeast = 0`

---

## Output Specification

### Prediction Output
- **Type**: Float
- **Currency**: Indian Rupees (₹)
- **Format**: Fixed decimal (2 places)
- **Range**: Model predicts values based on learned patterns
- **Display Format**: `💰 Predicted Insurance Charges: ₹{prediction:.2f}`

### Typical Output Ranges
- **Low Risk**: ₹1,000 - ₹5,000 (young, non-smoker, normal BMI)
- **Medium Risk**: ₹5,000 - ₹15,000 (middle-aged, healthy BMI)
- **High Risk**: ₹15,000 - ₹65,000+ (older, smoker, obese)

---

## Feature Matrix

### Final Features Used in Model

```
Feature Name              | Type        | Storage | Encoding
--------------------------|-------------|---------|------------------
age                       | Numeric     | Scaled  | StandardScaler
is_female                 | Binary      | Encoded | 0/1
bmi                       | Numeric     | Scaled  | StandardScaler
children                  | Numeric     | Scaled  | StandardScaler
is_smoker                 | Binary      | Encoded | 0/1
region_southeast          | Binary      | Encoded | 0/1
bmi_category_Obese        | Binary      | Encoded | 0/1
```

### Total Features: 7

---

## Data Flow & Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUTS (Streamlit)                   │
├─────────────────────────────────────────────────────────────┤
│ Age | Gender | BMI | Children | Smoker | Region             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │   FEATURE ENCODING       │
         ├──────────────────────────┤
         │ Gender → is_female      │
         │ Smoker → is_smoker      │
         │ Region → region_southeast│
         │ BMI → bmi_category_Obese │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │   FEATURE SCALING        │
         ├──────────────────────────┤
         │ age (StandardScaler)    │
         │ bmi (StandardScaler)    │
         │ children (StandardScaler)│
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │    FEATURE ORDERING      │
         ├──────────────────────────┤
         │ [age, is_female, bmi,   │
         │  children, is_smoker,   │
         │  region_southeast,      │
         │  bmi_category_Obese]    │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │  MODEL PREDICTION        │
         ├──────────────────────────┤
         │ Linear Regression Model  │
         │ Input: 7 features        │
         │ Output: Charge value     │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │  FORMAT OUTPUT           │
         ├──────────────────────────┤
         │ Display in ₹ with 2 DPs  │
         └──────────────────────────┘
```

---

## Model Interaction Details

### Loading Process
```python
model = joblib.load("Linear_Regression_model.pkl")      # 1. Load model
scaler = joblib.load("scaler.pkl")                      # 2. Load scaler
expected_columns = joblib.load("columns.pkl")           # 3. Load column order
```

### Prediction Process
```python
# 1. Create input DataFrame with encoded features
input_df = pd.DataFrame([input_data])

# 2. Ensure column order matches training
input_df = input_df[expected_columns]

# 3. Apply scaling to numeric columns
input_df_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

# 4. Make prediction
prediction = model.predict(input_df_scaled)[0]

# 5. Display result
st.success(f"💰 Predicted Insurance Charges: ₹{prediction:.2f}")
```

---

## Error Handling

### Validation Rules

| Field | Min | Max | Required | Type |
|-------|-----|-----|----------|------|
| age | 18 | 100 | Yes | int |
| bmi | 10 | 50 | Yes | float |
| children | 0 | 5 | Yes | int |
| gender | - | - | Yes | select |
| smoker | - | - | Yes | select |
| region | - | - | Yes | select |

### Error Messages (if implemented)
- "Age must be between 18 and 100"
- "BMI must be between 10 and 50"
- "Number of children must be 0-5"
- "Please select a valid option"

---

## Model Performance Metrics

### Training Set Performance
- **Samples**: 1,069 records
- **Features**: 7

### Test Set Performance
- **Samples**: 268 records
- **R² Score**: 0.8041 (80.41%)
- **Adjusted R²**: 0.7988 (79.88%)
- **Mean Absolute Error**: ~$4,500 (approx)

### Interpretation
- Model explains 80% of variance in test set
- Good generalization (train-test split consistent)
- Suitable for preliminary estimates, not replacing actual quotes

---

## Known Limitations

1. **Static Model**: Uses fixed coefficients learned during training
2. **Linear Assumptions**: Assumes linear relationships between features
3. **Limited Features**: Doesn't include health history, occupation
4. **Geographic**: Trained on US data, converted to INR
5. **Temporal**: Doesn't account for inflation/date changes
6. **No Interaction Terms**: Only main effects modeled
7. **Outlier Sensitivity**: Linear regression sensitive to extreme values

---

## Integration Notes

### Required Files in Same Directory
```
app.py
Linear_Regression_model.pkl
scaler.pkl
columns.pkl
```

### Environment Variables
None required (all paths are local)

### Dependencies
- streamlit >= 1.0.0
- pandas >= 1.0.0
- joblib >= 1.0.0
- scikit-learn >= 0.24.0

### Compatibility
- Python: 3.7+
- OS: Windows, Linux, macOS

---

## Example Usage Scenarios

### Scenario 1: Young, Healthy Non-Smoker
```
Input:  Age=25, Gender=Male, BMI=22, Children=0, Smoker=No, Region=Northwest
Output: Low prediction (estimated: ₹2,000-4,000)
```

### Scenario 2: Middle-Aged Smoker
```
Input:  Age=45, Gender=Female, BMI=32, Children=2, Smoker=Yes, Region=Southeast
Output: High prediction (estimated: ₹25,000-35,000)
```

### Scenario 3: Older Adult, Normal BMI
```
Input:  Age=60, Gender=Male, BMI=25, Children=3, Smoker=No, Region=Northeast
Output: Medium-High prediction (estimated: ₹12,000-18,000)
```

---

## Feature Importance Ranking

Based on Pearson Correlation with target (charges):

1. **is_smoker**: 0.787 ⭐⭐⭐⭐⭐ (Strongest)
2. **age**: 0.298 ⭐⭐⭐
3. **bmi_category_Obese**: 0.200 ⭐⭐
4. **bmi**: 0.196 ⭐⭐
5. **region_southeast**: 0.074 ⭐
6. **children**: 0.067 ⭐
7. **is_female**: -0.058 (Negative, minimal)

---

## Scaling Information

### StandardScaler Parameters (Fitted on Training Data)

```
Feature    | Mean   | Std Dev
-----------|--------|----------
age        | 39.21  | 14.05
bmi        | 30.66  | 6.10
children   | 1.09   | 1.21
```

### Scaling Formula
```
scaled_value = (value - mean) / std_dev
```

This ensures all numeric features have same scale (mean=0, std=1).

---

**Document Version**: 1.0
**Last Updated**: 2026-02-07
