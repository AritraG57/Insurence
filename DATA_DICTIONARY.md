# Data Dictionary & Feature Guide

## Complete Feature Reference

### Original Dataset (insurance.csv)

#### Raw Features (Before Processing)

| Column | Data Type | Null Count | Unique Values | Min | Max | Mean | Median |
|--------|-----------|-----------|---------------|-----|-----|------|--------|
| age | int64 | 0 | 47 | 18 | 64 | 39.21 | 39 |
| sex | object | 0 | 2 | - | - | - | - |
| bmi | float64 | 0 | 548 | 15.96 | 53.13 | 30.66 | 30.40 |
| children | int64 | 0 | 6 | 0 | 5 | 1.09 | 1 |
| smoker | object | 0 | 2 | - | - | - | - |
| region | object | 0 | 4 | - | - | - | - |
| charges | float64 | 0 | 1339 | 1121.87 | 63770.43 | 13270.42 | 9382.03 |

---

## Processed Features (Used in Model)

### Numeric Features (Continuous)

#### age
- **Original Name**: age
- **Data Type**: Integer
- **Range**: 18-64 years (training), 18-100 (predicted)
- **Processing**: StandardScaler normalization
- **Correlation with Charges**: +0.298 (positive)
- **Interpretation**: Older individuals have higher insurance charges
- **Example Values**: 18, 25, 40, 55, 64
- **Distribution**: Roughly uniform across age groups

#### bmi (Body Mass Index)
- **Original Name**: bmi
- **Data Type**: Float (scaled to 2 decimals)
- **Range**: 15.96 - 53.13 kg/m²
- **Processing**: StandardScaler normalization
- **Correlation with Charges**: +0.196 (weak positive)
- **Interpretation**: Higher BMI associated with higher charges
- **Example Values**: 16.0, 25.0, 30.0, 45.0, 53.0
- **BMI Categories**:
  - Underweight: BMI < 18.5
  - Normal Weight: 18.5 ≤ BMI < 25.0
  - Overweight: 25.0 ≤ BMI < 30.0
  - Obese: BMI ≥ 30.0
- **Distribution**: Slightly right-skewed, concentrated around 30

#### children (Number of Dependents)
- **Original Name**: children
- **Data Type**: Integer
- **Range**: 0-5 dependents
- **Processing**: StandardScaler normalization (though categorical nature)
- **Correlation with Charges**: +0.067 (very weak positive)
- **Interpretation**: Minimal impact on charges
- **Example Values**: 0, 1, 2, 3, 4, 5
- **Distribution**: Most common: 0 or 1 child
- **Value Counts**:
  - 0 children: ~573 records
  - 1 child: ~514 records
  - 2 children: ~163 records
  - 3+ children: ~87 records

---

### Binary/Categorical Features (Encoded)

#### is_female
- **Original Feature**: sex
- **Original Values**: "male", "female"
- **Encoded Values**: 0 (male), 1 (female)
- **Data Type**: Binary Integer
- **Correlation with Charges**: -0.058 (very weak negative)
- **Interpretation**: Gender has minimal impact; slight difference exists
- **Distribution**:
  - Male (0): 675 records (50.5%)
  - Female (1): 662 records (49.5%)
- **Balance**: Nearly perfectly balanced

#### is_smoker
- **Original Feature**: smoker
- **Original Values**: "no", "yes"
- **Encoded Values**: 0 (no), 1 (yes)
- **Data Type**: Binary Integer
- **Correlation with Charges**: +0.787 (very strong positive) ⭐⭐⭐⭐⭐
- **Interpretation**: STRONGEST PREDICTOR. Smokers pay significantly more.
- **Distribution**:
  - Non-smoker (0): 1,063 records (79.5%)
  - Smoker (1): 274 records (20.5%)
- **Impact**: Smokers typically pay 3-4x more in insurance

#### region_southeast
- **Original Feature**: region
- **Original/Dropped Values**: "northwest", "northeast", "southwest", "southeast"
- **Encoded Feature**: region_southeast (1 = southeast, 0 = other)
- **Data Type**: Binary Integer (one-hot encoded, drop_first=True)
- **Other Region Encoding**:
  - region_northwest: Dropped (baseline)
  - region_northeast: Implicitly No (0)
  - region_southwest: Implicitly No (0)
  - region_southeast: Explicitly Yes (1) or No (0)
- **Correlation with Charges**: +0.074 (very weak positive)
- **Interpretation**: Region has minimal impact; Southeast slightly higher
- **Distribution**:
  - Northwest (0): 324 records
  - Northeast (0): 324 records
  - Southwest (0): 325 records
  - Southeast (1): 364 records

#### bmi_category_Obese
- **Original Feature**: bmi (derived)
- **Derivation Method**: pd.cut() with bins [0, 18.5, 24.9, 29.9, inf]
- **Original Categories**: ['Underweight', 'Normal', 'Overweight', 'Obese']
- **Encoded Feature**: bmi_category_Obese (1 = obese, 0 = not obese)
- **Data Type**: Binary Integer (one-hot encoded, drop_first=True)
- **Dropped Categories**:
  - bmi_category_Normal: Dropped (baseline)
  - bmi_category_Underweight: Dropped (rare)
  - bmi_category_Overweight: Dropped
- **Correlation with Charges**: +0.200 (weak positive)
- **Interpretation**: Obesity status moderately predictive of higher charges
- **Distribution**:
  - Not Obese (0): ~950 records (71%)
  - Obese (1): ~387 records (29%)

---

## Target Variable

### charges (Insurance Premium)
- **Data Type**: Float (USD currency)
- **Range**: $1,121.87 - $63,770.43
- **Mean**: $13,270.42
- **Median**: $9,382.03
- **Std Dev**: $12,110.01
- **Interpretation**: Annual health insurance cost per individual
- **Distribution**: Right-skewed (long tail of high-cost individuals)
- **Quartiles**:
  - 25th percentile: $4,740.29
  - 50th percentile: $9,382.03
  - 75th percentile: $16,639.91
- **Data Quality**: No missing values, no outliers removed

---

## Feature Selection Results

### Chi-Square Test Results (Categorical Features)

| Feature | Chi² Statistic | P-Value | Decision | Interpretation |
|---------|-----------------|---------|----------|-----------------|
| is_smoker | 848.22 | 0.000 | ✓ KEEP | Highly significant |
| region_southeast | 15.99 | 0.001 | ✓ KEEP | Significant |
| is_female | 10.26 | 0.017 | ✓ KEEP | Significant |
| bmi_category_Obese | 8.52 | 0.036 | ✓ KEEP | Significant |
| region_southwest | 5.09 | 0.165 | ✗ DROP | Not significant |
| bmi_category_Overweight | 4.25 | 0.236 | ✗ DROP | Not significant |
| bmi_category_Normal | 3.71 | 0.295 | ✗ DROP | Not significant |
| region_northwest | 1.13 | 0.769 | ✗ DROP | Not significant |

**Significance Level**: α = 0.05

### Pearson Correlation Ranking (All Features)

| Rank | Feature | Correlation | Strength |
|------|---------|-------------|----------|
| 1️⃣ | is_smoker | 0.787 | ⭐⭐⭐⭐⭐ Very Strong |
| 2️⃣ | age | 0.298 | ⭐⭐⭐ Moderate |
| 3️⃣ | bmi_category_Obese | 0.200 | ⭐⭐ Weak |
| 4️⃣ | bmi | 0.196 | ⭐⭐ Weak |
| 5️⃣ | region_southeast | 0.074 | ⭐ Very Weak |
| 6️⃣ | children | 0.067 | ⭐ Very Weak |
| 7️⃣ | is_female | -0.058 | ⭐ Very Weak (Negative) |

---

## Preprocessing Pipeline Details

### Step 1: Data Loading
- **Source**: insurance.csv
- **Records Loaded**: 1,338
- **Columns**: 7

### Step 2: Duplicate Removal
- **Duplicates Found & Removed**: 1
- **Records After**: 1,337

### Step 3: Null Value Check
- **Result**: No null values found in any column

### Step 4: Categorical Encoding
- **Sex → is_female**: Direct mapping (0=male, 1=female)
- **Smoker → is_smoker**: Direct mapping (0=no, 1=yes)

### Step 5: One-Hot Encoding (Region)
- **Original Unique Values**: 4 regions
- **Dummy Variables Created**: 3 (drop_first=True)
- **Resulting Columns**:
  - region_northwest (dropped as baseline)
  - region_northeast (implicit, not shown)
  - region_southwest (redundant, later dropped)
  - region_southeast (kept)

### Step 6: Feature Engineering (BMI Categories)
- **Binning Thresholds**:
  ```
  Underweight: [0, 18.5)
  Normal:      [18.5, 24.9)
  Overweight:  [24.9, 29.9)
  Obese:       [29.9, ∞)
  ```
- **One-Hot Encoding**: Similar to region, only bmi_category_Obese kept

### Step 7: Feature Scaling
- **Scaler Type**: StandardScaler (z-score normalization)
- **Scaled Features**: age, bmi, children
- **Scaling Formula**: (x - mean) / std_dev
- **Expected Output**: Mean ≈ 0, Std Dev ≈ 1
- **Reference Values** (from training set):
  ```
  age:      mean=39.21, std=14.05
  bmi:      mean=30.66, std=6.10
  children: mean=1.09,  std=1.21
  ```

### Step 8: Final Feature Selection
- **Total Features Retained**: 7
- **Method**: Chi-square test + Pearson correlation analysis
- **Final Features**:
  1. age (scaled)
  2. is_female (encoded)
  3. bmi (scaled)
  4. children (scaled)
  5. is_smoker (encoded)
  6. region_southeast (encoded)
  7. bmi_category_Obese (encoded)

---

## Train/Test Split Details

### Data Split
- **Total Samples**: 1,337
- **Training Set**: 1,069 (80%)
- **Test Set**: 268 (20%)
- **Random State**: 42 (reproducible split)

### Train Set Statistics
| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| age | 39.5 | 14.2 | 19 | 63 |
| bmi | 30.7 | 6.1 | 16.1 | 52.8 |
| children | 1.1 | 1.2 | 0 | 5 |

### Test Set Statistics
| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| age | 38.5 | 13.7 | 18 | 64 |
| bmi | 30.5 | 6.1 | 16.0 | 53.1 |
| children | 1.0 | 1.2 | 0 | 5 |

---

## Data Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Missing Values | 0 | ✓ Good |
| Duplicates | 1 (removed) | ✓ Good |
| Outliers | None detected | ✓ Good |
| Class Balance (smoker) | 79.5% / 20.5% | ⚠️ Imbalanced |
| Feature Completeness | 100% | ✓ Perfect |
| Data Types | Correct | ✓ Good |

---

## Feature Interaction Notes

### Observed Patterns
1. **Smoker + Age**: Strong combined effect (older smokers pay most)
2. **Smoker + Obesity**: Combined effect is additive
3. **Gender + Smoker**: Gender effect minimal regardless of smoking
4. **Region**: Weak interaction with most features

### Not Modeled (Linear Regression Limitation)
- Feature interactions (e.g., age × smoker)
- Non-linear relationships
- High-order polynomial terms
- Complex categorical combinations

---

## Recommendations for Feature Usage

### When Making Predictions
1. **Age**: Use accurate current age
2. **BMI**: Calculate from height/weight or look up current value
3. **Smoker Status**: Use current smoking status (or intended status)
4. **Region**: Select actual residence region
5. **Gender**: Use biological sex as recorded
6. **Children**: Count all dependents

### Data Entry Tips
- **Age**: Integer, 18-100 years
- **BMI**: Calculate as weight(kg) / height(m)²
- **All fields together**: Provide complete profile for best accuracy

---

## Feature Sensitivity Analysis

### Impact of 1-Unit Change (Approximate)

| Feature | Change | Impact on Charges |
|---------|--------|-------------------|
| age | +1 year | +$250-300 |
| bmi | +1 point | +$50-100 |
| children | +1 child | +$150-200 |
| smoker | No → Yes | +$13,000-15,000 |
| gender | Male → Female | -$100-200 |
| region-Southeast | No → Yes | +$300-400 |
| obesity | Normal → Obese | +$3,000-4,000 |

*Note: Approximate values, actual impact depends on other features*

---

**Document Version**: 1.0
**Last Updated**: 2026-02-07
**Data Source**: Original insurance dataset (1,338 records)
