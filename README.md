# Task 1 – Data Cleaning & Preprocessing (Titanic Dataset)

## Objective
Clean and prepare raw Titanic data for machine-learning models.

## Steps Followed
1. **Import & Inspect**
   - Loaded CSV and checked info, null counts, and basic statistics.
2. **Handle Missing Values**
   - Filled `Age` with median.
   - Filled `Embarked` with mode.
   - Dropped `Cabin` (too many nulls).
3. **Encode Categorical Features**
   - One-hot encoded `Sex` and `Embarked`.
4. **Normalize/Standardize**
   - Standardized `Age` and `Fare` using `StandardScaler`.
5. **Detect & Remove Outliers**
   - Used boxplots and IQR method to remove extreme values.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

