# Binary Classification with Logistic Regression
## Breast Cancer Wisconsin Dataset Analysis

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements a **binary classification model** using **Logistic Regression** to predict whether a breast cancer tumor is **malignant** or **benign** based on various tumor characteristics. The project demonstrates a complete machine learning pipeline from data exploration to model evaluation and visualization.

### 🎯 Objective
Build a robust binary classifier that can accurately distinguish between malignant and benign breast cancer tumors using the Wisconsin Breast Cancer Dataset.

## 📊 Dataset Information

- **Dataset**: Wisconsin Breast Cancer Dataset
- **Samples**: 569 tumor samples
- **Features**: 30 numerical features (after preprocessing)
- **Target Classes**: 
  - **Malignant (M)**: 212 cases (37.3%)
  - **Benign (B)**: 357 cases (62.7%)

### Feature Categories
The dataset contains three types of features for each tumor:
1. **Mean values** (e.g., radius_mean, texture_mean)
2. **Standard error** (e.g., radius_se, texture_se) 
3. **Worst values** (e.g., radius_worst, texture_worst)

## 🛠️ Installation & Requirements

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🚀 Usage

### Running the Complete Analysis
```bash
python TASK4.PY
```

This will execute the entire pipeline including:
- Data loading and exploration
- Preprocessing and cleaning
- Exploratory data analysis
- Model training and evaluation
- Feature importance analysis
- Prediction examples
- Visualization generation

### Generated Output Files
- `eda_analysis.png` - Exploratory Data Analysis visualizations
- `model_evaluation.png` - Model performance metrics and plots
- `feature_importance.png` - Feature importance analysis

## 📈 Results Summary

### Model Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 96.49% |
| **ROC AUC** | 99.60% |
| **Precision (Benign)** | 96% |
| **Precision (Malignant)** | 97% |
| **Recall (Benign)** | 99% |
| **Recall (Malignant)** | 93% |
| **F1-Score (Benign)** | 97% |
| **F1-Score (Malignant)** | 95% |

### Top 5 Most Important Features
1. **texture_worst** (coefficient: 1.43)
2. **radius_se** (coefficient: 1.23)
3. **symmetry_worst** (coefficient: 1.06)
4. **concave_points_mean** (coefficient: 0.95)
5. **concavity_worst** (coefficient: 0.91)

## 📁 Project Structure

```
TASK 4/archive (3)/
├── TASK4.PY                    # Main analysis script
├── data.csv                    # Wisconsin Breast Cancer Dataset
├── README.md                   # Project documentation
├── eda_analysis.png           # Exploratory data analysis plots
├── model_evaluation.png       # Model evaluation visualizations
└── feature_importance.png     # Feature importance analysis
```

## 🔬 Methodology

### 1. Data Preprocessing
- Removed unnecessary columns (ID, unnamed columns)
- Handled missing values
- Encoded target variable (M=1, B=0)
- Feature standardization using StandardScaler

### 2. Exploratory Data Analysis
- Target variable distribution analysis
- Feature correlation analysis
- Box plots for key features by diagnosis
- Feature-target correlation analysis

### 3. Model Training
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 80-20 stratified split
- **Feature Scaling**: StandardScaler normalization
- **Hyperparameters**: max_iter=1000, random_state=42

### 4. Model Evaluation
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curve**: Area Under Curve analysis
- **Precision-Recall Curve**: Performance on imbalanced data
- **Classification Report**: Detailed metrics by class

### 5. Feature Analysis
- **Coefficient Analysis**: Logistic regression coefficients
- **Feature Importance**: Ranking by absolute coefficient values
- **Direction Analysis**: Positive/negative impact on classification

## 📊 Visualizations

### 1. Exploratory Data Analysis (`eda_analysis.png`)
- Target variable distribution (pie chart)
- Feature correlation heatmap
- Box plots comparing features by diagnosis
- Top correlated features with target

### 2. Model Evaluation (`model_evaluation.png`)
- Confusion matrix heatmap
- ROC curve with AUC score
- Precision-Recall curve
- Prediction probability distributions

### 3. Feature Importance (`feature_importance.png`)
- Horizontal bar chart of top 15 features
- Coefficient values and directions
- Color-coded positive/negative impacts

## 🎯 Key Insights

1. **Excellent Performance**: The model achieves 96.49% accuracy with 99.60% ROC AUC
2. **Feature Significance**: Texture and shape-related features are most predictive
3. **Class Balance**: Despite slight class imbalance, the model performs well on both classes
4. **Generalization**: High precision and recall suggest good generalization capability

## 🔍 Model Interpretation

- **Positive Coefficients**: Features that increase probability of malignancy
- **Negative Coefficients**: Features that increase probability of benignancy
- **Texture Features**: Most influential in classification decisions
- **Shape Irregularities**: Strong indicators of malignant tumors

## 🚦 Usage Examples

### Making Predictions on New Data
```python
# Load the trained model and scaler
# model, scaler = load_trained_model()

# Prepare new sample data
# new_sample = preprocess_new_data(sample_data)

# Make prediction
# prediction = model.predict(scaler.transform(new_sample))
# probability = model.predict_proba(scaler.transform(new_sample))
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Source**: UCI Machine Learning Repository
- **Original Research**: Wisconsin Breast Cancer Dataset creators
- **Tools**: Scikit-learn, Pandas, Matplotlib, Seaborn

## 📧 Contact

For questions or suggestions regarding this project, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. For medical applications, please consult with healthcare professionals and ensure proper validation and regulatory compliance. 