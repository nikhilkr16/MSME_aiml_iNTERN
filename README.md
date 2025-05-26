🚢 MSME_AIML - Task 1: Titanic Dataset Preprocessing
This repository showcases the preprocessing pipeline applied to the Titanic dataset as part of MSME AI/ML training. It demonstrates exploratory data analysis, handling of missing values, encoding, feature scaling, and feature selection to prepare the data for machine learning tasks.

📌 Key Features
🧾 1. Dataset Overview
Displays basic dataset information: number of rows, columns, and missing values.

Identifies and categorizes features:

Numerical: Age, Fare, SibSp, Parch

Categorical: Sex, Embarked

Text: Cabin (converted to a binary feature)

❓ 2. Missing Value Analysis
Age: Filled with the mean age.

Embarked: Filled with the most frequent embarkation port.

Cabin: Converted to a new feature HasCabin indicating presence (1) or absence (0) of cabin info.

🛠️ Preprocessing Pipeline
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load and explore data
df = pd.read_csv('titanic.csv')
print(df.info())
print(df.isnull().sum())

# Step 2: Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['HasCabin'] = df['Cabin'].notna().astype(int)

# Step 3: Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'])

# Step 4: Scale numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 5: Handle outliers (optional)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Step 6: Select final features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 
            'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin']
X = df[features]
y = df['Survived']

print("Dataset is ready for machine learning!")
print(f"Shape: {X.shape}")
print(f"Features: {list(X.columns)}")
🖼️ Sample Outputs
Step	Screenshot
Data Info & Missing Values	
Filled Missing Values	
Cabin Feature Processing	
Encoding Categorical Variables	
Scaling Numerical Features	
Final Output	

📂 Dataset
Source: Titanic Dataset

Columns used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Cabin, Survived

✅ Output
Cleaned and processed feature matrix X

Target label vector y

Ready for training machine learning models like Logistic Regression, Random Forest, etc.
