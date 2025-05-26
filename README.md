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

| Description           | Screenshot                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| Data Info             | ![Screenshot 1](https://github.com/user-attachments/assets/a41853be-3317-49c7-b59b-7ed7bcb397b6) |
| Filled Missing Values | ![Screenshot 2](https://github.com/user-attachments/assets/0b8281fa-7d25-4115-84fa-fa18b2ce9646) |
| Cabin Feature         | ![Screenshot 3](https://github.com/user-attachments/assets/fefb533a-0cb1-4c33-8ec1-3eef5a4eca85) |
| Encoding Categorical  | ![Screenshot 4](https://github.com/user-attachments/assets/6a8666a3-5aa9-4692-bd04-66cfd9509aef) |
| Feature Scaling       | ![Screenshot 5](https://github.com/user-attachments/assets/96a35222-58ee-4c1a-896f-70d42ed2c766) |
| Final Output          | ![Screenshot 6](https://github.com/user-attachments/assets/2088ea51-faf7-497f-9ab8-4bcf6fc2fe48) |

## 📁 Dataset Source
- [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

✅ Output
X: Final feature matrix (ready for ML models)

y: Target variable (Survived)

Shape and features printed after processing


---

📬 Contact
Feel free to raise an issue or connect for questions, feedback, or collaborations.
### ✅ How to Use:
1. Copy this into your `README.md` file.
2. Update the `username/repo-name` part of image links if needed (based on actual repo name or upload structure).
3. You can also add badges (e.g., Python version, last update) if you'd like a more professional look.

 

