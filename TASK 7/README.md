# Task 7: Support Vector Machines (SVM) Analysis

## Objective
Use Support Vector Machines (SVMs) for both linear and non-linear classification on the Breast Cancer Wisconsin dataset.

## Dataset
- **File**: `breast-cancer.csv`
- **Description**: Breast Cancer Wisconsin dataset containing features computed from digitized images of fine needle aspirate (FNA) of breast masses
- **Target**: Diagnosis (M = Malignant, B = Benign)
- **Features**: 30 numerical features describing characteristics of cell nuclei

## Features
- **Linear SVM**: Support Vector Machine with linear kernel
- **Non-linear SVMs**: RBF, Polynomial, and Sigmoid kernels
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Model Evaluation**: Comprehensive performance metrics
- **Visualization**: Decision boundaries, ROC curves, and feature importance
- **Data Analysis**: EDA with correlation analysis and PCA

## Tools Used
- **Scikit-learn**: SVM implementation and model evaluation
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation
- **Seaborn**: Statistical visualization

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python main.py
```

## What the Analysis Includes

### 1. Data Exploration
- Dataset overview and statistics
- Missing value analysis
- Target variable distribution
- Feature correlation analysis

### 2. Data Preprocessing
- Feature scaling with StandardScaler
- Label encoding for target variable
- Train-test split with stratification

### 3. Visualization
- Target distribution plots
- Feature correlation heatmap
- PCA visualization
- Feature distribution by diagnosis

### 4. SVM Models

#### Linear SVM
- Uses linear kernel for linearly separable data
- Hyperparameter tuning for regularization parameter C
- Feature importance analysis through weights

#### Non-linear SVMs
- **RBF (Radial Basis Function)**: For complex, non-linear patterns
- **Polynomial**: For polynomial relationships in data
- **Sigmoid**: For neural network-like decision boundaries

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Classification reports
- Confusion matrices
- ROC curves and AUC scores
- Cross-validation analysis

### 6. Advanced Visualizations
- Decision boundary visualization (2D PCA space)
- Support vector highlighting
- Model performance comparison
- Feature importance analysis

## Key SVM Concepts Demonstrated

### Support Vector Machines
- **Support Vectors**: Data points closest to the decision boundary
- **Margin**: Distance between decision boundary and nearest data points
- **Kernel Trick**: Mapping data to higher dimensions for non-linear separation

### Kernels Explained
1. **Linear**: `K(x,y) = x·y` - For linearly separable data
2. **RBF**: `K(x,y) = exp(-γ||x-y||²)` - For complex non-linear patterns
3. **Polynomial**: `K(x,y) = (γx·y + r)^d` - For polynomial relationships
4. **Sigmoid**: `K(x,y) = tanh(γx·y + r)` - Similar to neural networks

### Hyperparameters
- **C**: Regularization parameter (controls overfitting)
- **gamma**: Kernel coefficient for RBF, polynomial, and sigmoid
- **degree**: Polynomial degree for polynomial kernel

## Expected Results

The analysis will demonstrate:
- High classification accuracy (typically >95%) on this dataset
- Comparison between linear and non-linear approaches
- Feature importance insights for medical diagnosis
- Visual understanding of decision boundaries

## Medical Significance

This analysis shows how SVMs can be applied to:
- Medical diagnosis and classification
- Feature importance for understanding disease characteristics
- Model interpretability in healthcare applications
- Robust classification with limited data

## Output

The script generates comprehensive output including console results and multiple visualizations:

### Console Output
Detailed analysis results including:

#### Dataset Overview
```
Dataset shape: (569, 32)
Target variable distribution:
B    357  (Benign cases)
M    212  (Malignant cases)
Missing values: 0
Feature matrix shape: (569, 30)
Training set shape: (455, 30)
Test set shape: (114, 30)
```

#### Model Training Results
```
LINEAR SVM:
Best parameters: {'C': 0.01, 'kernel': 'linear'}
Cross-validation score: 96.92% (+/- 4.48%)

RBF SVM:
Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
Cross-validation score: 97.58%

POLYNOMIAL SVM:
Best parameters: {'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
Cross-validation score: 94.07%

SIGMOID SVM:
Best parameters: {'C': 1, 'gamma': 0.01, 'kernel': 'sigmoid'}
Cross-validation score: 96.48%
```

#### Detailed Performance Metrics
For each model, the console displays:
- Accuracy, Precision, Recall, F1-score
- Complete classification reports with per-class metrics
- Confusion matrices with true/false positive/negative counts
- Cross-validation scores with standard deviation

#### Sample Classification Report (RBF SVM - Best Performer)
```
RBF SVM Results:
Accuracy: 0.9737
Precision: 1.0000
Recall: 0.9286
F1-score: 0.9630

Classification Report:
              precision    recall  f1-score   support
      Benign       0.96      1.00      0.98        72
   Malignant       1.00      0.93      0.96        42
    accuracy                           0.97       114
   macro avg       0.98      0.96      0.97       114
weighted avg       0.97      0.97      0.97       114

Confusion Matrix:
[[72  0]    # 72 True Negatives, 0 False Positives
 [ 3 39]]   # 3 False Negatives, 39 True Positives
```

#### PCA Analysis Output
```
PCA explained variance ratio: [0.44593522 0.18545255]
Total variance explained by first 2 components: 63.14%
```

### Visualizations Generated

#### 1. Data Exploration Visualizations (Figure 1: 20x15 inches)
**Subplot Layout: 2 rows × 3 columns**

- **Target Distribution (Bar Chart)**
  - Shows count of Benign (357) vs Malignant (212) cases
  - Color-coded: Light coral for Benign, Light blue for Malignant
  - Reveals class imbalance in the dataset

- **Feature Correlation Heatmap**
  - 11×11 correlation matrix of key features plus diagnosis
  - Features: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean
  - Color scale: Cool-warm colormap showing correlations from -1 to +1
  - Reveals strong correlations between size-related features

- **PCA Visualization (2D Scatter Plot)**
  - Data points projected onto first two principal components
  - Color-coded by diagnosis (Viridis colormap: 0=Benign, 1=Malignant)
  - Shows 63.14% total variance explained (44.59% + 18.55%)
  - Demonstrates good class separation in reduced dimensional space

- **Feature Distribution Histograms (2 plots)**
  - Overlaid histograms for radius_mean and texture_mean
  - Separate distributions for Benign (B) and Malignant (M) cases
  - Shows clear differences in feature distributions between classes

#### 2. Model Performance Comparison (Figure 2: 20x15 inches)
**Subplot Layout: 3 rows × 3 columns**

- **Individual Metric Bar Charts (4 plots)**
  - Accuracy Comparison: RBF (97.37%) > Linear/Sigmoid (95.61%) > Polynomial (93.86%)
  - Precision Comparison: All models achieve perfect 100% precision
  - Recall Comparison: RBF (92.86%) > Linear/Sigmoid (88.10%) > Polynomial (83.33%)
  - F1-Score Comparison: RBF (96.30%) > Linear/Sigmoid (93.67%) > Polynomial (90.91%)

- **Combined Metrics Bar Chart**
  - Side-by-side comparison of all 4 metrics for all models
  - Color-coded bars for easy visual comparison
  - Shows RBF SVM's superiority across most metrics

- **ROC Curves Comparison**
  - ROC curves for all 4 SVM models
  - AUC scores displayed in legend
  - Diagonal reference line for random classifier
  - All models show excellent performance (AUC > 0.95)

#### 3. Confusion Matrices (Figure 3: 16x12 inches)
**Subplot Layout: 2 rows × 2 columns**

- **Linear SVM**: [[72, 0], [5, 37]] - 5 false negatives, 0 false positives
- **RBF SVM**: [[72, 0], [3, 39]] - 3 false negatives, 0 false positives (best)
- **Polynomial SVM**: [[72, 0], [7, 35]] - 7 false negatives, 0 false positives
- **Sigmoid SVM**: [[72, 0], [5, 37]] - 5 false negatives, 0 false positives

Heat map style with blue color scale, annotated with actual counts

#### 4. Decision Boundaries Visualization (Figure 4: 20x15 inches)
**Subplot Layout: 2 rows × 2 columns**

- **2D Decision Boundaries in PCA Space**
  - Each subplot shows decision boundary for different kernel
  - Contour plots showing classification regions
  - Training data points overlaid (color-coded by class)
  - Support vectors highlighted with red circles
  - Demonstrates how different kernels create different decision boundaries:
    - Linear: Straight line boundary
    - RBF: Smooth, curved boundary
    - Polynomial: More complex curved boundary
    - Sigmoid: S-shaped boundary

#### 5. Feature Importance Analysis (Figure 5: 12x8 inches)
**Subplot Layout: 1 row × 2 columns**

- **Top 15 Feature Importance (Horizontal Bar Chart)**
  - Ranked by absolute weight values from Linear SVM
  - Shows most discriminative features for cancer diagnosis
  - Helps understand which measurements are most critical

- **All Feature Weights (Vertical Bar Chart)**
  - Complete weight distribution across all 30 features
  - Shows relative importance of all features
  - Useful for feature selection and model interpretation

### Performance Summary Table
```
Model          | Accuracy | Precision | Recall | F1-Score | Cross-Val
---------------|----------|-----------|--------|----------|----------
RBF SVM        | 97.37%   | 100.00%   | 92.86% | 96.30%   | 97.58%
Linear SVM     | 95.61%   | 100.00%   | 88.10% | 93.67%   | 96.92%
Sigmoid SVM    | 95.61%   | 100.00%   | 88.10% | 93.67%   | 96.48%
Polynomial SVM | 93.86%   | 100.00%   | 83.33% | 90.91%   | 94.07%
```

### Key Visual Insights
1. **Perfect Precision**: All models achieve 100% precision (no false positives)
2. **RBF Superiority**: RBF SVM shows best overall performance
3. **Class Separation**: PCA visualization shows good natural class separation
4. **Feature Correlations**: Strong correlations exist between size-related features
5. **Decision Boundaries**: Non-linear kernels capture complex patterns better
6. **Medical Relevance**: Zero false positives crucial for medical applications

## Complete Visualization Summary

When you run `python main.py`, you will see exactly **5 comprehensive figures** displaying:

### 📊 Figure 1: Data Exploration Dashboard
- **Purpose**: Understanding the dataset characteristics
- **Layout**: 6 subplots showing data distribution, correlations, and PCA projections
- **Key Insights**: Class imbalance, feature relationships, dimensionality reduction effectiveness

### 📈 Figure 2: Model Performance Analysis
- **Purpose**: Comparing all SVM models across multiple metrics
- **Layout**: 6 subplots with individual and combined metric comparisons plus ROC curves
- **Key Insights**: RBF SVM superiority, perfect precision across all models

### 🔥 Figure 3: Confusion Matrices Heatmaps
- **Purpose**: Detailed error analysis for each model
- **Layout**: 4 heatmaps (one per SVM variant)
- **Key Insights**: All models avoid false positives, RBF minimizes false negatives

### 🎯 Figure 4: Decision Boundaries Visualization
- **Purpose**: Understanding how different kernels separate data
- **Layout**: 4 contour plots in 2D PCA space
- **Key Insights**: Visual representation of linear vs non-linear decision making

### 📋 Figure 5: Feature Importance Analysis
- **Purpose**: Identifying most critical features for diagnosis
- **Layout**: 2 plots showing top features and complete weight distribution
- **Key Insights**: Which measurements are most important for cancer detection

### 📝 Console Analytics
- **Real-time progress tracking** during model training
- **Hyperparameter optimization results** with GridSearchCV
- **Detailed performance metrics** with statistical confidence intervals
- **Cross-validation scores** ensuring model robustness

This comprehensive visualization suite provides complete insight into SVM performance for medical diagnosis, making it suitable for both educational purposes and practical medical applications.

## Files Structure
```
TASK 7/
├── main.py              # Main SVM analysis script
├── breast-cancer.csv    # Dataset
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Learning Outcomes

By running this analysis, you'll understand:
- How SVMs work for classification tasks
- Difference between linear and non-linear kernels
- Hyperparameter tuning strategies
- Model evaluation techniques
- Feature importance in medical datasets
- Visualization of high-dimensional data 