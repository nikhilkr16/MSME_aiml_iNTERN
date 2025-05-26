# 🚢 MSME_AIML - Titanic Dataset Preprocessing (Task 1)

This repository contains the preprocessing pipeline for the Titanic dataset, developed as part of the MSME AI/ML training program. The goal is to clean and prepare the dataset for machine learning tasks by handling missing values, encoding categorical variables, scaling numerical features, and selecting relevant features.

---

## 📌 Key Features

### 🧾 Dataset Overview
- Displays structure and summary of the dataset.
- Identifies types of features:
  - **Numerical**: Age, Fare, SibSp, Parch
  - **Categorical**: Sex, Embarked
  - **Text**: Cabin (transformed to binary)

### ❓ Missing Value Analysis
- `Age`: Filled with the **mean**.
- `Embarked`: Filled with the **most frequent value**.
- `Cabin`: Converted to a binary feature `HasCabin`.

---

## 🛠️ Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and explore data
df = pd.read_csv('titanic.csv')
print(df.info())
print(df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['HasCabin'] = df['Cabin'].notna().astype(int)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Handle outliers in Fare (optional)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Final feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male',
            'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin']
X = df[features]
y = df['Survived']

print("Dataset is ready for machine learning!")
print(f"Shape: {X.shape}")
print(f"Features: {list(X.columns)}")
