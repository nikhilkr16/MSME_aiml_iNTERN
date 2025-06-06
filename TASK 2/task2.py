import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Agg')

# Read the dataset
df = pd.read_csv('archive/Titanic-Dataset.csv')

# 1. Basic Summary Statistics
print("\n=== Basic Summary Statistics ===")
print("\nNumerical Columns Summary:")
print(df.describe())

print("\nCategorical Columns Summary:")
print(df.describe(include=['object']))

# 2. Missing Values Analysis
print("\n=== Missing Values Analysis ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 3. Visualizations
plt.style.use('default')
plt.figure(figsize=(15, 10))

# Create a 2x2 subplot for basic distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age Distribution
sns.histplot(data=df, x='Age', bins=30, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')

# Fare Distribution
sns.histplot(data=df, x='Fare', bins=30, ax=axes[0,1])
axes[0,1].set_title('Fare Distribution')

# Survival Count
sns.countplot(data=df, x='Survived', ax=axes[1,0])
axes[1,0].set_title('Survival Count')

# Passenger Class Distribution
sns.countplot(data=df, x='Pclass', ax=axes[1,1])
axes[1,1].set_title('Passenger Class Distribution')

plt.tight_layout()
plt.savefig('basic_distributions.png')
plt.close()

# 4. Correlation Analysis
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 5. Advanced Visualizations using Plotly
# Age vs Fare scatter plot with survival information
fig = px.scatter(df, x='Age', y='Fare', color='Survived',
                 title='Age vs Fare by Survival Status',
                 labels={'Age': 'Age', 'Fare': 'Fare'},
                 color_discrete_sequence=['red', 'green'])
fig.write_html('age_fare_survival.html')

# Survival rate by passenger class
survival_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
fig = px.bar(survival_by_class, x='Pclass', y='Survived',
             title='Survival Rate by Passenger Class',
             labels={'Pclass': 'Passenger Class', 'Survived': 'Survival Rate'})
fig.write_html('survival_by_class.html')

# 6. Box Plots for Age and Fare by Passenger Class
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age Distribution by Passenger Class')

plt.subplot(1, 2, 2)
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Passenger Class')

plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()

# 7. Additional Insights
print("\n=== Additional Insights ===")
print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())

print("\nAverage Fare by Passenger Class:")
print(df.groupby('Pclass')['Fare'].mean())

print("\nSurvival Rate by Passenger Class:")
print(df.groupby('Pclass')['Survived'].mean())

# Save the processed data
df.to_csv('titanic_processed.csv', index=False)
