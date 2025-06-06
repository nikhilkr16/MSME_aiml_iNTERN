# AI/ML Internship Tasks - Comprehensive Overview (Tasks 1-8)

This repository contains a complete machine learning journey from data preprocessing to advanced algorithms. Each task builds upon previous concepts while introducing new techniques and methodologies.

## 📋 Table of Contents

- [Task 1: Data Preprocessing Pipeline](#task-1-data-preprocessing-pipeline)
- [Task 2: Exploratory Data Analysis (EDA)](#task-2-exploratory-data-analysis-eda)
- [Task 3: Linear Regression Analysis](#task-3-linear-regression-analysis)
- [Task 4: Logistic Regression Classification](#task-4-logistic-regression-classification)
- [Task 5: Decision Trees & Random Forests](#task-5-decision-trees--random-forests)
- [Task 6: K-Nearest Neighbors (KNN)](#task-6-k-nearest-neighbors-knn)
- [Task 7: Support Vector Machines (SVM)](#task-7-support-vector-machines-svm)
- [Task 8: K-Means Clustering](#task-8-k-means-clustering)
- [Technologies & Tools](#technologies--tools)
- [Learning Progression](#learning-progression)
- [Getting Started](#getting-started)

---

## Task 1: Data Preprocessing Pipeline

**📂 Directory**: `TASK 1/`  
**🎯 Objective**: Build a comprehensive data preprocessing pipeline for the Titanic dataset

### Key Features
- **Dataset**: Titanic passenger data (891 passengers)
- **Missing Value Handling**: Systematic approach to data cleaning
- **Feature Engineering**: Creating new meaningful features
- **Data Transformation**: Encoding categorical variables and scaling numerical features

### Technologies Used
- **Pandas** - Data manipulation and cleaning
- **NumPy** - Numerical operations
- **Scikit-learn** - StandardScaler for feature normalization

### Preprocessing Pipeline

#### 1. **Missing Value Treatment**
- **Age**: Filled with mean value (logical for continuous variable)
- **Embarked**: Filled with most frequent value (mode for categorical)
- **Cabin**: Transformed to binary feature `HasCabin` (presence/absence indicator)

#### 2. **Feature Engineering**
- **HasCabin**: Binary indicator (0/1) for cabin information availability
- **Categorical Encoding**: One-hot encoding for Sex and Embarked
- **Feature Selection**: Retained most predictive features

#### 3. **Data Transformation**
- **Standardization**: Applied StandardScaler to numerical features
- **Outlier Handling**: Used IQR method for fare outlier detection
- **Feature Scaling**: Normalized Age, Fare, SibSp, Parch

### Final Feature Set
```python
Features: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male',
           'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin']
Target: 'Survived'
```

### Key Insights
- **Data Quality**: Systematic handling of 19.9% missing age values
- **Feature Engineering**: Cabin presence proved to be a valuable predictor
- **Encoding Strategy**: One-hot encoding preserved categorical information
- **Scaling Impact**: Standardization essential for distance-based algorithms

### Preprocessing Results
- **Original Shape**: (891, 12) → **Processed Shape**: (889, 10) features
- **Missing Values**: Eliminated from all selected features
- **Feature Types**: All numerical (ready for ML algorithms)
- **Data Integrity**: Maintained passenger relationships and survival patterns

### Business Value
- **Data Quality**: Clean, consistent dataset for downstream analysis
- **Feature Engineering**: Enhanced predictive power through thoughtful transformations
- **Scalability**: Reusable preprocessing pipeline for similar datasets
- **ML Readiness**: Formatted data optimized for machine learning algorithms

---

## Task 2: Exploratory Data Analysis (EDA)

**📂 Directory**: `TASK 2/`  
**🎯 Objective**: Master data exploration techniques using the famous Titanic dataset

### Key Features
- **Dataset**: Titanic passenger data (891 passengers)
- **Survival Rate**: 38.4% overall survival
- **Comprehensive EDA**: Statistical analysis, missing value handling, and pattern identification
- **Visualizations**: Static plots (PNG) and interactive visualizations (HTML)

### Technologies Used
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Static plotting and visualizations  
- **Seaborn** - Advanced statistical visualizations
- **Plotly** - Interactive visualizations

### Key Insights
- Gender had significant impact on survival rates
- Passenger class strongly correlated with survival chances
- Age and fare distributions revealed socioeconomic patterns
- Missing data analysis guided preprocessing strategies

### Generated Outputs
- `basic_distributions.png` - Age, fare, survival, class distributions
- `correlation_matrix.png` - Feature relationship heatmap
- `boxplots.png` - Class-based statistical distributions
- `age_fare_survival.html` - Interactive scatter plot
- `survival_by_class.html` - Interactive survival analysis

---

## Task 3: Linear Regression Analysis

**📂 Directory**: `TASK 3/`  
**🎯 Objective**: Implement simple and multiple linear regression for housing price prediction

### Key Features
- **Dataset**: Housing price data (545 houses, 13 features)
- **Price Range**: $1.75M - $13.3M
- **Model Comparison**: Simple vs Multiple Linear Regression
- **Feature Engineering**: Categorical encoding and correlation analysis

### Performance Results
| Model Type | R² Score | RMSE | MAE | Improvement |
|-----------|----------|------|-----|-------------|
| Simple Linear | 28.9% | $1.9M | $1.5M | Baseline |
| Multiple Linear | 64.9% | $1.3M | $980K | +125.2% |

### Key Findings
- **Most Important Features**: Bathrooms (+$1.1M), A/C (+$786K), Hot Water (+$688K)
- **Area Impact**: Each sq ft adds $614.68 to house price
- **Location Premium**: Preferred areas command ~$630K premium
- **Model Evolution**: Multiple features dramatically improved prediction accuracy

### Generated Outputs
- `housing_regression_analysis.png` - Model comparison visualization
- `Figure_1.png` - Additional regression analysis

---

## Task 4: Logistic Regression Classification

**📂 Directory**: `TASK 4/archive (3)/`  
**🎯 Objective**: Binary classification for breast cancer diagnosis

### Key Features
- **Dataset**: Wisconsin Breast Cancer Dataset (569 samples, 30 features)
- **Classes**: Malignant (37.3%) vs Benign (62.7%)
- **High Accuracy**: 96.49% classification accuracy
- **Feature Analysis**: Comprehensive importance ranking

### Performance Metrics
- **Accuracy**: 96.49%
- **ROC AUC**: 99.60%
- **Precision**: Benign (96%), Malignant (97%)
- **Recall**: Benign (99%), Malignant (93%)

### Top Predictive Features
1. **texture_worst** (coefficient: 1.43)
2. **radius_se** (coefficient: 1.23)
3. **symmetry_worst** (coefficient: 1.06)
4. **concave_points_mean** (coefficient: 0.95)
5. **concavity_worst** (coefficient: 0.91)

### Generated Outputs
- `eda_analysis.png` - Exploratory data analysis
- `model_evaluation.png` - Performance metrics visualization
- `feature_importance.png` - Feature ranking analysis

---

## Task 5: Decision Trees & Random Forests

**📂 Directory**: `TASK 5/`  
**🎯 Objective**: Learn tree-based models for classification and regression

### Key Features
- **Dataset**: Heart Disease Prediction (13 medical features)
- **Model Types**: Decision Trees and Random Forest ensembles
- **Medical Application**: Clinical decision support system
- **Hyperparameter Optimization**: Grid search for optimal parameters

### Model Comparison
- **Decision Tree**: ~75-85% accuracy, high interpretability
- **Random Forest**: ~80-90% accuracy, better generalization
- **Key Predictors**: Chest pain type, max heart rate, ST depression

### Medical Insights
- **Risk Factors**: Chest pain type strongest predictor
- **Exercise Capacity**: Lower max heart rate indicates risk
- **Clinical Value**: Feature importance guides medical decisions
- **Overfitting Analysis**: Demonstrated ensemble benefits

### Generated Outputs
- Tree visualization plots
- Feature importance comparisons
- Learning curve analysis
- Cross-validation results

---

## Task 6: K-Nearest Neighbors (KNN)

**📂 Directory**: `TASK 6/`  
**🎯 Objective**: Implement KNN classification with comprehensive optimization

### Key Features
- **Dataset**: Iris flower classification (150 samples, 4 features, 3 classes)
- **Perfect Balance**: 50 samples per class
- **Optimization**: Cross-validation for optimal k selection
- **Advanced Analysis**: PCA visualization and decision boundaries

### Performance Results
- **Final Accuracy**: 95.6%
- **Optimal k-value**: 14 (determined via 5-fold cross-validation)
- **Best CV Score**: 98.1%
- **PCA Variance**: 96.01% explained by first two components

### Key Findings
- **Perfect Separation**: Iris-setosa completely separable
- **Feature Importance**: Petal measurements most discriminative
- **Model Stability**: High accuracy with excellent generalization
- **Visualization**: Clear decision boundaries in 2D PCA space

### Generated Outputs
- `iris_exploratory_analysis.png` - Data exploration dashboard
- `optimal_k_selection.png` - K-value optimization curve
- `model_evaluation.png` - Performance metrics
- `decision_boundary.png` - 2D decision boundary visualization

---

## Task 7: Support Vector Machines (SVM)

**📂 Directory**: `TASK 7/`  
**🎯 Objective**: Master SVM for linear and non-linear classification

### Key Features
- **Dataset**: Breast Cancer Wisconsin (569 samples, 30 features)
- **Multiple Kernels**: Linear, RBF, Polynomial, Sigmoid
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Advanced Visualizations**: Decision boundaries, support vectors

### Model Performance
| Kernel | Cross-Validation Score | Best Parameters |
|--------|----------------------|-----------------|
| **RBF** | 97.58% | C=1, gamma='scale' |
| **Linear** | 96.92% | C=0.01 |
| **Sigmoid** | 96.48% | C=1, gamma=0.01 |
| **Polynomial** | 94.07% | C=10, degree=3 |

### Key Concepts Demonstrated
- **Support Vector Theory**: Margin maximization
- **Kernel Trick**: Non-linear data transformation
- **Feature Importance**: Medical diagnosis insights
- **Model Interpretability**: Clinical decision support

### Generated Outputs
- `data_exploration.png` - Comprehensive EDA dashboard
- `model_performance.png` - Performance comparison
- `confusion_matrices.png` - Detailed classification results
- `decision_boundaries.png` - 2D boundary visualization
- `feature_importance.png` - Feature ranking analysis

---

## Task 8: K-Means Clustering

**📂 Directory**: `TASK 8/`  
**🎯 Objective**: Customer segmentation using unsupervised learning

### Key Features
- **Dataset**: Mall Customer data (200 customers, 5 features)
- **Features**: Age, Annual Income, Spending Score, Gender
- **Optimal Clusters**: 6 segments (via Elbow Method + Silhouette Analysis)
- **Business Application**: Marketing strategy development

### Customer Segments Identified
1. **High Earners, High Spenders** (19.5%) - Premium targets
2. **Young High Spenders** (11.5%) - Trendy product focus
3. **High Earners, Low Spenders** (16.5%) - Value-conscious
4. **Middle-aged Moderate Spenders** (22.5%) - Balanced approach
5. **Young Average Earners** (19.5%) - Standard marketing
6. **Older Low Earners** (10.5%) - Budget-friendly options

### Business Value
- **Targeted Marketing**: Customized campaigns per segment
- **Product Strategy**: Tailored recommendations
- **Pricing Optimization**: Segment-based pricing
- **Customer Retention**: Specific retention strategies

### Generated Outputs
- `optimal_clusters_analysis.png` - Cluster selection methodology
- `clustering_results_2d.png` - Multiple 2D perspectives
- `clustering_results_3d.png` - 3D comprehensive visualization

---

## 🛠️ Technologies & Tools

### Core Libraries
- **Python 3.7+** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing foundation
- **Scikit-learn** - Machine learning algorithms and utilities

### Visualization
- **Matplotlib** - Static plotting and visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations

### Specialized Tools
- **StandardScaler** - Feature normalization
- **GridSearchCV** - Hyperparameter optimization
- **Cross-validation** - Model evaluation
- **PCA** - Dimensionality reduction

---

## 📈 Learning Progression

### Phase 1: Foundation (Tasks 1-3)
- **Data Preprocessing**: Cleaning, transforming, and preparing data
- **Data Exploration**: Understanding datasets through EDA
- **Statistical Analysis**: Descriptive statistics and correlations
- **Basic Modeling**: Linear regression concepts
- **Visualization Skills**: Creating meaningful plots

### Phase 2: Classification (Tasks 4-7)
- **Binary Classification**: Logistic regression implementation
- **Tree-based Methods**: Decision trees and ensemble learning
- **Distance-based Learning**: KNN algorithm and optimization
- **Kernel Methods**: SVM with multiple kernel types

### Phase 3: Advanced Techniques (Task 8)
- **Unsupervised Learning**: Clustering without labels
- **Business Applications**: Real-world problem solving
- **Customer Analytics**: Practical marketing insights
- **Optimization Methods**: Finding optimal parameters

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
Git
```

### Installation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AIML-INTERN
   ```

2. **Navigate to specific task:**
   ```bash
   cd "TASK X"  # Replace X with task number
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis:**
   ```bash
   python main.py  # or specific task file
   ```

### Quick Start Guide
- **Complete Beginners**: Start with Task 1 (Preprocessing) to learn data preparation
- **Data Analysis Focus**: Move to Task 2 (EDA) to understand data exploration
- **Intermediate**: Jump to Task 5 (Tree Methods) for practical ML
- **Advanced**: Focus on Task 7 (SVM) for algorithmic depth
- **Business Focus**: Task 8 (Clustering) for practical applications

---

## 📊 Datasets Summary

| Task | Dataset | Samples | Features | Domain | Target |
|------|---------|---------|----------|--------|--------|
| 1 | Titanic | 891 | 12→10 | Transportation | Survival |
| 2 | Titanic | 891 | 12 | Transportation | Survival |
| 3 | Housing | 545 | 13 | Real Estate | Price |
| 4 | Breast Cancer | 569 | 30 | Medical | Diagnosis |
| 5 | Heart Disease | 1027 | 13 | Medical | Disease |
| 6 | Iris | 150 | 4 | Botanical | Species |
| 7 | Breast Cancer | 569 | 30 | Medical | Diagnosis |
| 8 | Mall Customers | 200 | 5 | Business | Segments |

---

## 🎯 Key Learning Outcomes

### Technical Skills
✅ **Data Preprocessing**: Cleaning, encoding, scaling  
✅ **Exploratory Data Analysis**: Statistical insights and visualizations  
✅ **Supervised Learning**: Classification and regression algorithms  
✅ **Unsupervised Learning**: Clustering and dimensionality reduction  
✅ **Model Evaluation**: Metrics, cross-validation, hyperparameter tuning  
✅ **Feature Engineering**: Selection, importance, and transformation  

### Business Applications
✅ **Medical Diagnosis**: Clinical decision support systems  
✅ **Real Estate**: Price prediction and market analysis  
✅ **Customer Analytics**: Segmentation and targeting strategies  
✅ **Risk Assessment**: Survival analysis and risk factors  

### Visualization & Communication
✅ **Static Plotting**: Publication-ready charts and graphs  
✅ **Interactive Dashboards**: User-friendly data exploration  
✅ **Business Reporting**: Actionable insights and recommendations  
✅ **Technical Documentation**: Comprehensive project documentation  

---

## 📝 License

This project is part of an AI/ML internship program and demonstrates various machine learning techniques on real-world datasets. All code is available for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Contact

For questions or suggestions about any of the tasks, please open an issue in this repository.

---

**Note**: This comprehensive overview showcases a complete machine learning journey from basic data analysis to advanced algorithms, providing both theoretical understanding and practical implementation skills. 
