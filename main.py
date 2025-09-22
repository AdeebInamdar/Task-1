import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Titanic-Dataset.csv")
print("\n--- Basic Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())


# Handle missing values (mean/median/imputation)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
# print(df)


# 3️Convert categorical features into numerical using encoding
# One-hot encode 'Sex' and 'Embarked'
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Normalize/standardize the numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print(df)


# Visualize outliers using boxplots and remove them
# Save boxplots before removing outliers (optional)
plt.figure(figsize=(8,4))
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age (Standardized)")
plt.savefig("age_boxplot.png")
plt.close()

plt.figure(figsize=(8,4))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare (Standardized)")
plt.savefig("fare_boxplot.png")
plt.close()

# Remove outliers using IQR
for col in ['Age', 'Fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Save the cleaned dataset
df.to_csv("titanic_task1_cleaned.csv", index=False)

print("\n--- Cleaning Complete ---")
print("Cleaned dataset saved as titanic_task1_cleaned.csv")
print("\n--- Final Preview ---")
print(df.head())





















