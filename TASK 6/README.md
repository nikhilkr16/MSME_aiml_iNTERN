# Task 6: K-Nearest Neighbors (KNN) Classification - Complete Implementation

## 🎯 Objective
Understand and implement KNN for classification problems using the classic Iris dataset. This project demonstrates a comprehensive machine learning workflow from data exploration to model evaluation.

## 🛠️ Tools & Technologies
- **Scikit-learn** - Machine learning algorithms and utilities
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **NumPy** - Numerical computing

## 📊 Dataset Information
- **Dataset**: Iris flower classification dataset
- **Total samples**: 150
- **Features**: 4 numerical features
  - SepalLengthCm
  - SepalWidthCm  
  - PetalLengthCm
  - PetalWidthCm
- **Classes**: 3 iris species
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Class balance**: Perfectly balanced (50 samples each)

## 🏗️ Project Structure

```
TASK 6/
├── main.py                           # Interactive KNN implementation
├── main_non_interactive.py           # Non-interactive version (saves plots)
├── requirements.txt                  # Project dependencies
├── Iris.csv                         # Dataset file
├── analysis_report.txt              # Comprehensive analysis summary
├── iris_exploratory_analysis.png    # Data exploration visualizations
├── optimal_k_selection.png          # K-value optimization plot
├── model_evaluation.png             # Model performance visualizations
├── decision_boundary.png            # Decision boundary visualization
└── README.md                        # This documentation
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Interactive Version**
   ```bash
   python main.py
   ```

3. **Run Non-Interactive Version** (saves all plots)
   ```bash
   python main_non_interactive.py
   ```

## 📈 Implementation Features

### 1. **Data Exploration & Visualization**
- Comprehensive exploratory data analysis (EDA)
- Feature distribution analysis by species
- Correlation matrix of features
- Scatter plots for feature relationships
- Box plots showing statistical distributions

### 2. **Data Preprocessing**
- Feature scaling using StandardScaler
- Train-test split with stratification (70-30 split)
- Proper handling of categorical target labels
- Missing value analysis

### 3. **Model Optimization**
- Cross-validation for optimal k selection
- Grid search for hyperparameter tuning
- Multiple distance metrics comparison (Euclidean, Manhattan, Minkowski)
- Weight schemes evaluation (uniform vs distance-based)

### 4. **Model Evaluation**
- Detailed classification report with precision, recall, F1-score
- Confusion matrix visualization
- Prediction confidence analysis
- Cross-validation scoring

### 5. **Advanced Analysis**
- PCA for dimensionality reduction and visualization
- Decision boundary visualization in 2D space
- Component analysis interpretation
- Comprehensive reporting and documentation

## 📊 Key Results & Performance

### **Model Performance Metrics**

```
============================================================
IRIS DATASET - K-NEAREST NEIGHBORS CLASSIFICATION
============================================================

Final Model Accuracy: 95.6%

Classification Report:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       0.88      1.00      0.94        15
 Iris-virginica       1.00      0.87      0.93        15

       accuracy                           0.96        45
      macro avg       0.96      0.96      0.96        45
   weighted avg       0.96      0.96      0.96        45
```

### **Optimization Results**

- **Optimal k-value**: 14 (determined through 5-fold cross-validation)
- **Best cross-validation score**: 98.1%
- **Best hyperparameters**: 
  - n_neighbors: 14
  - weights: 'uniform'
  - metric: 'euclidean'
- **PCA variance explained**: 96.01% (first two components)

### **Key Findings**

1. **Feature Importance**: Petal length and width are the most discriminative features
2. **Class Separability**: Iris-setosa is perfectly separable from other classes
3. **Model Robustness**: High accuracy with excellent generalization
4. **Optimal Configuration**: k=14 with uniform weights and Euclidean distance

## 📸 Generated Visualizations

### 1. **iris_exploratory_analysis.png**
- **Petal vs Sepal Analysis**: Scatter plots showing species clustering
- **Feature Distributions**: Box plots revealing statistical properties
- **Correlation Matrix**: Heatmap showing feature relationships
- **Species Separation**: Visual confirmation of class separability

### 2. **optimal_k_selection.png**
- **K-value Optimization Curve**: Cross-validation accuracy vs k
- **Optimal Point Identification**: Clear visualization of k=14 as optimal
- **Performance Stability**: Shows model behavior across different k values

### 3. **model_evaluation.png**
- **Confusion Matrix**: Detailed breakdown of predictions vs actual
- **Prediction Confidence**: Scatter plot showing model certainty
- **Error Analysis**: Visual identification of misclassified samples

### 4. **decision_boundary.png**
- **2D Decision Boundaries**: PCA projection showing classification regions
- **Species Clustering**: Visual representation of class separation
- **Component Analysis**: Feature contribution to principal components

## 📋 Detailed Analysis Output

```
1. DATASET OVERVIEW:
------------------------------
Dataset shape: (150, 6)
Features: ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
Target classes: ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
Class distribution:
Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50

Missing values: 0

3. DATA PREPARATION:
------------------------------
Features shape: (150, 4)
Target shape: (150,)
Training set: 105 samples
Test set: 45 samples
✓ Features scaled using StandardScaler

4. FINDING OPTIMAL K:
------------------------------
Optimal k: 14
Best cross-validation score: 0.9810

5. MODEL TRAINING AND EVALUATION:
------------------------------
Test Accuracy: 0.9556

6. DECISION BOUNDARY VISUALIZATION:
------------------------------
Total variance explained by 2 components: 96.01%

7. HYPERPARAMETER TUNING:
------------------------------
Best parameters: {'metric': 'euclidean', 'n_neighbors': 14, 'weights': 'uniform'}
Best cross-validation score: 0.9810

8. FINAL MODEL PERFORMANCE:
------------------------------
Final model accuracy: 0.9556
```

## 🔍 Code Structure & Implementation

### **Main Functions**

1. **`load_and_explore_data()`**
   - Dataset loading and initial exploration
   - Statistical summary and missing value analysis
   - Class distribution visualization

2. **`visualize_data(df)`**
   - Comprehensive EDA with multiple plot types
   - Feature relationship analysis
   - Statistical distribution visualization

3. **`prepare_data(df)`**
   - Feature-target separation
   - Train-test split with stratification
   - Feature scaling and normalization

4. **`find_optimal_k(X_train_scaled, y_train)`**
   - Cross-validation across k values (1-20)
   - Performance curve generation
   - Optimal k identification

5. **`train_and_evaluate_knn(...)`**
   - Model training with optimal parameters
   - Comprehensive evaluation metrics
   - Visualization of results

6. **`visualize_decision_boundary(...)`**
   - PCA dimensionality reduction
   - 2D decision boundary plotting
   - Component analysis

7. **`hyperparameter_tuning(...)`**
   - Grid search optimization
   - Multiple parameter combinations
   - Best model selection

## 📊 Technical Specifications

### **Model Configuration**
- **Algorithm**: K-Nearest Neighbors
- **Distance Metric**: Euclidean
- **Weight Scheme**: Uniform
- **Number of Neighbors**: 14
- **Cross-Validation**: 5-fold stratified

### **Data Processing**
- **Scaling Method**: StandardScaler (z-score normalization)
- **Train-Test Split**: 70-30 stratified split
- **Random State**: 42 (for reproducibility)

### **Evaluation Metrics**
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class prediction accuracy
- **Recall**: Per-class detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## 🎯 Key Insights & Recommendations

### **Model Performance**
- ✅ **Excellent Accuracy**: 95.6% test accuracy demonstrates strong performance
- ✅ **Balanced Performance**: Good precision and recall across all classes
- ✅ **Robust Generalization**: High cross-validation scores indicate good generalization

### **Feature Analysis**
- 🔍 **Petal Features**: Most discriminative for species classification
- 🔍 **Feature Scaling**: Beneficial for KNN performance
- 🔍 **Low Dimensionality**: 4 features provide sufficient information

### **Algorithm Suitability**
- ✅ **Perfect for Iris Dataset**: KNN works exceptionally well
- ✅ **Optimal k Selection**: k=14 provides best balance of bias-variance
- ✅ **Distance Metric**: Euclidean distance performs optimally

### **Future Enhancements**
- 🚀 **Ensemble Methods**: Consider Random Forest or SVM for comparison
- 🚀 **Feature Engineering**: Explore feature combinations or transformations
- 🚀 **Cross-Dataset Validation**: Test on other botanical datasets
- 🚀 **Real-time Classification**: Implement web interface for live predictions

## 🎓 Learning Outcomes

This implementation demonstrates:

1. **Complete ML Pipeline**: From data exploration to model deployment
2. **Best Practices**: Proper validation, scaling, and evaluation techniques  
3. **Hyperparameter Optimization**: Systematic approach to model tuning
4. **Visualization Skills**: Comprehensive plotting and analysis
5. **Documentation**: Professional code documentation and reporting

## 📝 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🏆 Conclusion

This KNN implementation showcases a comprehensive machine learning workflow that achieves excellent performance on the Iris classification task. The 95.6% accuracy, combined with detailed analysis and visualization, demonstrates both technical proficiency and practical understanding of the K-Nearest Neighbors algorithm.

The project serves as an excellent learning resource for understanding:
- KNN algorithm mechanics and parameter tuning
- Data preprocessing and feature scaling importance
- Model evaluation and validation techniques
- Professional ML project documentation and visualization

---

**Author**: AIML Intern Project  
**Date**: June 2025  
**Version**: 1.0 