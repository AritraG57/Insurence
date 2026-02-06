# Insurance Price Prediction System - Documentation

## Overview

The Insurance Price Prediction System is a machine learning application that predicts insurance charges based on individual characteristics. It uses a Linear Regression model trained on insurance data to estimate charges for customers in Indian Rupees (₹).

## Table of Contents

- [Project Structure](#project-structure)
- [Data Overview](#data-overview)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Setup and Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)
- [Technical Implementation](#technical-implementation)

---

## Project Structure

```
Insurence/
├── app.py                              # Streamlit web application
├── model.ipynb                         # Jupyter notebook with ML pipeline
├── Linear_Regression_model.pkl         # Trained model (serialized)
├── scaler.pkl                          # StandardScaler for feature normalization
├── columns.pkl                         # Expected feature columns
└── insurance.csv                       # Original dataset (referenced in notebook)
```

---

## Data Overview

### Dataset Information

- **Total Records**: 1,338 (after removing 1 duplicate)
- **Features**: 7 original features
- **Target Variable**: `charges` (insurance costs in USD)

### Original Features

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| age | Integer | Age of the individual | 18-64 years |
| sex | Categorical | Gender | Male, Female |
| bmi | Float | Body Mass Index | 15.96 - 53.13 |
| children | Integer | Number of children covered | 0-5 |
| smoker | Categorical | Smoking status | Yes, No |
| region | Categorical | Geographic region | NW, NE, SW, SE |
| charges | Float | Insurance charges | $1,121.87 - $63,770.43 |

### Statistical Summary

| Statistic | Age | BMI | Children | Charges |
|-----------|-----|-----|----------|---------|
| Mean | 39.21 | 30.66 | 1.09 | $13,270.42 |
| Median | 39 | 30.40 | 1 | $9,382.03 |
| Std Dev | 14.05 | 6.10 | 1.21 | $12,110.01 |
| Min | 18 | 15.96 | 0 | $1,121.87 |
| Max | 64 | 53.13 | 5 | $63,770.43 |

### Data Quality

- **Missing Values**: None (0 null values)
- **Duplicates**: 1 duplicate removed during preprocessing
- **Data Distribution**: Balanced across regions
  - Southeast: 364 records
  - Southwest: 325 records
  - Northwest: 324 records
  - Northeast: 324 records

---

## Machine Learning Pipeline

### 1. Data Loading & Exploration

- Load insurance data using pandas
- Perform exploratory data analysis (EDA) with visualizations
- Generate histograms, box plots, and correlation heatmap
- Analyze distributions of categorical and numerical features

### 2. Data Preprocessing

#### Encoding Categorical Variables

- **Sex**: Mapped to binary (Male=0, Female=1)
- **Smoker**: Mapped to binary (No=0, Yes=1)
- **Region**: One-hot encoded (3 dummy variables created, first dropped)
  - region_northwest
  - region_southeast
  - region_southwest

#### Feature Engineering

- **BMI Category Creation**: Created categorical bins for BMI
  - Underweight: BMI < 18.5
  - Normal: 18.5 ≤ BMI < 24.9
  - Overweight: 24.9 ≤ BMI < 29.9
  - Obese: BMI ≥ 30.0

- **One-hot Encoding for BMI Categories**: Converted categories to dummy variables
  - bmi_category_Normal
  - bmi_category_Overweight
  - bmi_category_Obese

### 3. Feature Scaling

Applied StandardScaler normalization to numeric features:
- **Scaled Features**: age, bmi, children
- **Scaling Method**: Standard Normalization (z-score)
- **Formula**: (x - mean) / std_dev

### 4. Feature Selection

#### Pearson Correlation Analysis (Numeric Features)

Top features by correlation with charges:

| Feature | Correlation | Interpretation |
|---------|-------------|-----------------|
| is_smoker | 0.787 | Very Strong Positive |
| age | 0.298 | Weak Positive |
| bmi_category_Obese | 0.200 | Weak Positive |
| bmi | 0.196 | Weak Positive |
| region_southeast | 0.074 | Very Weak Positive |
| children | 0.067 | Very Weak Positive |

#### Chi-Square Test (Categorical Features)

Statistical significance test at α = 0.05:

| Feature | Chi² Stat | P-Value | Decision |
|---------|-----------|---------|----------|
| is_smoker | 848.22 | 0.000 | **Keep** ✓ |
| region_southeast | 15.99 | 0.001 | **Keep** ✓ |
| is_female | 10.26 | 0.017 | **Keep** ✓ |
| bmi_category_Obese | 8.52 | 0.036 | **Keep** ✓ |
| region_southwest | 5.09 | 0.165 | Drop |
| bmi_category_Overweight | 4.25 | 0.236 | Drop |
| bmi_category_Normal | 3.71 | 0.295 | Drop |
| region_northwest | 1.13 | 0.769 | Drop |

#### Final Feature Set (7 features)

1. age (scaled)
2. is_female
3. bmi (scaled)
4. children (scaled)
5. is_smoker
6. region_southeast
7. bmi_category_Obese

### 5. Model Training

- **Algorithm**: Linear Regression
- **Train-Test Split**: 80-20 (1,069 training, 268 test samples)
- **Random State**: 42 (for reproducibility)

### 6. Model Evaluation

**Performance Metrics**:
- **R² Score**: 0.804 (80.4% variance explained)
- **Adjusted R²**: 0.799 (adjusted for model complexity)

**Interpretation**: The model explains approximately 80% of the variation in insurance charges, indicating a good fit.

---

## Model Details

### Linear Regression Model

**Type**: Ordinary Least Squares (OLS) Linear Regression

**Features Used**: 7 features from preprocessing pipeline

**Model Output**: Predicted insurance charges in USD

### Model Coefficients

The model learns relationships between features and insurance charges during training. Key insights from feature importance:

- **Smoking Status**: Most significant predictor (highest correlation: 0.787)
- **Age**: Second most important factor (0.298)
- **BMI Category**: Moderate importance (0.200)
- **Region**: Minimal but statistically significant impact

### Serialized Model Files

1. **Linear_Regression_model.pkl**: Trained model object
2. **scaler.pkl**: StandardScaler instance for feature normalization
3. **columns.pkl**: List of expected feature columns in correct order

---

## Web Application

### Streamlit Application (`app.py`)

An interactive web interface for insurance charge prediction.

#### User Input Fields

| Input | Component | Range | Default |
|-------|-----------|-------|---------|
| Age | Slider | 18-100 years | 40 |
| Gender | Dropdown | Male, Female | Male |
| BMI | Slider | 10.0-50.0 | 24.0 |
| Children | Number Input | 0-5 | 0 |
| Smoker | Dropdown | Yes, No | No |
| Region | Dropdown | NW, NE, SW, SE | - |

#### Data Processing Flow

```
User Input
    ↓
Feature Encoding
    ├─ Gender → is_female (0/1)
    ├─ Smoker → is_smoker (0/1)
    ├─ Region → region_southeast (0/1)
    └─ BMI → bmi_category_Obese (0/1)
    ↓
Feature Scaling (age, bmi, children)
    ↓
Model Prediction
    ↓
Display Result
```

#### Feature Encoding Logic

**Gender Encoding**:
```
Male   → is_female = 0
Female → is_female = 1
```

**Smoker Encoding**:
```
No  → is_smoker = 0
Yes → is_smoker = 1
```

**Region Encoding**:
```
Region Southeast → region_southeast = 1
Other Regions    → region_southeast = 0
```

**BMI Category Encoding**:
```
BMI < 18.5           → bmi_category_Obese = 0
18.5 ≤ BMI < 25     → bmi_category_Obese = 0
25 ≤ BMI < 30       → bmi_category_Obese = 0
BMI ≥ 30            → bmi_category_Obese = 1
```

#### Output

Displays predicted insurance charge in Indian Rupees (₹) with 2 decimal places.

---

## Setup and Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Required Libraries

```
streamlit
pandas
joblib
scikit-learn
numpy
seaborn
matplotlib
scipy
```

### Installation Steps

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install streamlit pandas joblib scikit-learn numpy seaborn matplotlib scipy
```

3. **Navigate to project directory**:
```bash
cd "d:\My Programs\ML edit\Insurence"
```

4. **Ensure model files exist**:
   - `Linear_Regression_model.pkl`
   - `scaler.pkl`
   - `columns.pkl`

---

## Usage Guide

### Running the Web Application

```bash
streamlit run app.py
```

This will:
- Start a local Streamlit server
- Open the application in your default web browser
- Display at `http://localhost:8501/`

### Making Predictions

1. **Adjust sliders and dropdowns** for your characteristics
2. **Click "Predict Insurance Charges"** button
3. **View the predicted charge** displayed on screen

### Example Prediction

**Input**:
- Age: 35
- Gender: Female
- BMI: 28.5
- Children: 2
- Smoker: No
- Region: Northeast

**Expected Output**: ₹[predicted_amount]

---

## Technical Implementation

### Model Loading and Prediction (`app.py`)

```python
# Load pre-trained components
model = joblib.load("Linear_Regression_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")
```

### Input Data Processing

1. **Collect user inputs** from Streamlit widgets
2. **Encode categorical variables** to numeric values
3. **Create DataFrame** with input data
4. **Reorder columns** to match training order
5. **Scale numeric features** using fitted scaler
6. **Pass to model** for prediction

### Scaling Process

```python
scale_cols = ['age', 'bmi', 'children']
input_df_scaled[scale_cols] = scaler.transform(input_df[scale_cols])
```

**Important**: Only numeric features selected during training are scaled, preserving consistency with the model.

### Column Order Requirement

The model expects features in this exact order:
1. age
2. is_female
3. bmi
4. children
5. is_smoker
6. region_southeast
7. bmi_category_Obese

Failure to maintain this order will result in incorrect predictions.

---

## Key Insights & Recommendations

### Model Insights

1. **Smoking is the strongest predictor**: Smokers likely have significantly higher charges
2. **Age matters**: Older individuals tend to have higher premiums
3. **BMI impact**: Obese individuals have higher insurance costs
4. **Regional differences**: Southeast region shows slightly higher charges
5. **Gender and children**: Minimal individual impact on charges

### Recommendations for Users

1. **Smoking cessation**: Quitting smoking can significantly reduce insurance costs
2. **Weight management**: Maintaining healthy BMI (< 25) helps lower premiums
3. **Younger adoption**: Obtaining insurance at younger ages results in lower lifetime costs
4. **Regional considerations**: Regional differences exist but are minor

### Model Limitations

1. **Linear relationship assumption**: Real relationships may be non-linear
2. **Limited geographic scope**: Model trained on US-based data
3. **Fixed time period**: Model doesn't account for inflation or temporal changes
4. **Categorical feature gaps**: Some region combinations not fully captured
5. **External factors**: Doesn't consider pre-existing conditions, occupation, etc.

---

## File Dependencies

| File | Purpose | Required |
|------|---------|----------|
| Linear_Regression_model.pkl | Trained ML model | ✓ Yes |
| scaler.pkl | Feature scaling object | ✓ Yes |
| columns.pkl | Feature column order | ✓ Yes |
| insurance.csv | Raw training data | No* |

*Not required for predictions, only needed for model retraining

---

## Troubleshooting

### Common Issues

**Issue**: "FileNotFoundError: Linear_Regression_model.pkl"
- **Solution**: Ensure all .pkl files are in the same directory as app.py. Run model.ipynb if files are missing.

**Issue**: "ValueError: X has 8 features but model expects 7"
- **Solution**: Feature encoding logic may be incorrect. Check that all categorical variables are properly encoded.

**Issue**: Predictions seem unrealistic
- **Solution**: Verify that input data is within expected ranges (Age: 18-100, BMI: 10-50). Scaling is applied automatically.

**Issue**: "ModuleNotFoundError: No module named 'streamlit'"
- **Solution**: Install required packages: `pip install -r requirements.txt`

---

## Future Enhancements

1. **Model Improvement**:
   - Try ensemble methods (Random Forest, Gradient Boosting)
   - Implement feature interactions
   - Explore polynomial regression

2. **Feature Expansion**:
   - Include health history variables
   - Add occupation information
   - Consider family medical history

3. **Application Features**:
   - Batch prediction from CSV upload
   - Historical tracking of predictions
   - Risk assessment classification
   - Sensitivity analysis visualization

4. **Deployment**:
   - Deploy to cloud platforms (AWS, GCP, Azure)
   - Add authentication and user management
   - API endpoint development
   - Database integration for prediction logging

---

## Contact & Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Document Version**: 1.0
**Last Updated**: 2026-02-07
**Status**: Complete
