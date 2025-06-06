#!/usr/bin/env python3
"""
Task 7: Support Vector Machines (SVM)
Objective: Use SVMs for linear and non-linear classification
Tools: Scikit-learn, NumPy, Matplotlib
Dataset: breast-cancer.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_curve, auc)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SVMClassifier:
    """Class to implement and compare linear and non-linear SVM classifiers"""
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Models
        self.linear_svm = None
        self.rbf_svm = None
        self.poly_svm = None
        self.sigmoid_svm = None
        
    def load_and_explore_data(self):
        """Load and explore the dataset"""
        print("="*60)
        print("LOADING AND EXPLORING BREAST CANCER DATA")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nDataset info:")
        print(self.df.info())
        
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        print(f"\nTarget variable distribution:")
        print(self.df['diagnosis'].value_counts())
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum().sum())
        
        print(f"\nBasic statistics:")
        print(self.df.describe())
        
    def preprocess_data(self):
        """Preprocess the data for SVM"""
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # Prepare features and target
        # Drop id column and separate features from target
        feature_columns = [col for col in self.df.columns if col not in ['id', 'diagnosis']]
        self.X = self.df[feature_columns].values
        
        # Encode target variable (M=1, B=0)
        self.y = self.label_encoder.fit_transform(self.df['diagnosis'])
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target vector shape: {self.y.shape}")
        print(f"Class mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
    def visualize_data(self):
        """Visualize the data distribution and correlations"""
        print("\n" + "="*60)
        print("DATA VISUALIZATION")
        print("="*60)
        
        plt.figure(figsize=(20, 15))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        self.df['diagnosis'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
        plt.title('Distribution of Diagnosis')
        plt.xlabel('Diagnosis (B=Benign, M=Malignant)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Feature correlation heatmap (subset of features for visibility)
        plt.subplot(2, 3, 2)
        important_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                            'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
        
        # Convert diagnosis to numeric for correlation
        df_corr = self.df[important_features].copy()
        df_corr['diagnosis_numeric'] = self.label_encoder.fit_transform(self.df['diagnosis'])
        correlation_matrix = df_corr.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, cbar_kws={'shrink': .8})
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # PCA visualization for 2D plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_train_scaled)
        
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, 
                            cmap='viridis', alpha=0.7)
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization of Training Data')
        plt.colorbar(scatter, label='Diagnosis (0=Benign, 1=Malignant)')
        
        # Distribution of some key features
        key_features = ['radius_mean', 'texture_mean']
        for i, feature in enumerate(key_features):
            plt.subplot(2, 3, 4 + i)
            for diagnosis in ['B', 'M']:
                data = self.df[self.df['diagnosis'] == diagnosis][feature]
                plt.hist(data, alpha=0.7, label=f'{diagnosis}', bins=20)
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {feature}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained by first 2 components: {sum(pca.explained_variance_ratio_):.2%}")
        
    def train_linear_svm(self):
        """Train linear SVM classifier"""
        print("\n" + "="*60)
        print("TRAINING LINEAR SVM")
        print("="*60)
        
        # Grid search for best parameters
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear']
        }
        
        grid_search = GridSearchCV(
            SVC(random_state=42), param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.linear_svm = grid_search.best_estimator_
        
        print(f"Best parameters for Linear SVM: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.linear_svm, self.X_train_scaled, self.y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def train_nonlinear_svms(self):
        """Train non-linear SVM classifiers with different kernels"""
        print("\n" + "="*60)
        print("TRAINING NON-LINEAR SVMs")
        print("="*60)
        
        # RBF Kernel SVM
        print("\nTraining RBF SVM...")
        param_grid_rbf = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
        
        grid_search_rbf = GridSearchCV(
            SVC(random_state=42), param_grid_rbf, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search_rbf.fit(self.X_train_scaled, self.y_train)
        self.rbf_svm = grid_search_rbf.best_estimator_
        
        print(f"Best parameters for RBF SVM: {grid_search_rbf.best_params_}")
        print(f"Best CV score: {grid_search_rbf.best_score_:.4f}")
        
        # Polynomial Kernel SVM
        print("\nTraining Polynomial SVM...")
        param_grid_poly = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'kernel': ['poly']
        }
        
        grid_search_poly = GridSearchCV(
            SVC(random_state=42), param_grid_poly, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search_poly.fit(self.X_train_scaled, self.y_train)
        self.poly_svm = grid_search_poly.best_estimator_
        
        print(f"Best parameters for Polynomial SVM: {grid_search_poly.best_params_}")
        print(f"Best CV score: {grid_search_poly.best_score_:.4f}")
        
        # Sigmoid Kernel SVM
        print("\nTraining Sigmoid SVM...")
        param_grid_sigmoid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['sigmoid']
        }
        
        grid_search_sigmoid = GridSearchCV(
            SVC(random_state=42), param_grid_sigmoid, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search_sigmoid.fit(self.X_train_scaled, self.y_train)
        self.sigmoid_svm = grid_search_sigmoid.best_estimator_
        
        print(f"Best parameters for Sigmoid SVM: {grid_search_sigmoid.best_params_}")
        print(f"Best CV score: {grid_search_sigmoid.best_score_:.4f}")
        
    def evaluate_models(self):
        """Evaluate all SVM models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        models = {
            'Linear SVM': self.linear_svm,
            'RBF SVM': self.rbf_svm,
            'Polynomial SVM': self.poly_svm,
            'Sigmoid SVM': self.sigmoid_svm
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name} Results:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.decision_function(self.X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Benign', 'Malignant']))
            
            print(f"Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
            
        return results
        
    def visualize_results(self, results):
        """Visualize model comparison and results"""
        print("\n" + "="*60)
        print("VISUALIZING RESULTS")
        print("="*60)
        
        plt.figure(figsize=(20, 15))
        
        # Model comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(results.keys())
        
        for i, metric in enumerate(metrics):
            plt.subplot(3, 3, i + 1)
            values = [results[model][metric] for model in models]
            bars = plt.bar(models, values, color=['blue', 'green', 'red', 'orange'])
            plt.title(f'{metric.capitalize()} Comparison')
            plt.ylabel(metric.capitalize())
            plt.ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
        
        # Overall metrics comparison
        plt.subplot(3, 3, 5)
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            plt.bar(x + i * width, values, width, label=metric.capitalize())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('All Metrics Comparison')
        plt.xticks(x + width * 1.5, models, rotation=45)
        plt.legend()
        plt.ylim(0, 1.1)
        
        # ROC Curves
        plt.subplot(3, 3, 6)
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        
        # Note: Confusion matrices will be displayed separately
        pass  # Skip confusion matrices in this figure
        
        plt.tight_layout()
        plt.show()
        
        # Display confusion matrices separately
        plt.figure(figsize=(16, 12))
        for i, (model_name, result) in enumerate(results.items()):
            plt.subplot(2, 2, i + 1)
            cm = confusion_matrix(self.y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malignant'],
                       yticklabels=['Benign', 'Malignant'])
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_decision_boundaries(self):
        """Visualize decision boundaries using PCA-transformed data"""
        print("\n" + "="*60)
        print("VISUALIZING DECISION BOUNDARIES")
        print("="*60)
        
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(self.X_train_scaled)
        X_test_pca = pca.transform(self.X_test_scaled)
        
        # Train models on PCA data
        models_2d = {
            'Linear SVM': SVC(kernel='linear', C=1, random_state=42),
            'RBF SVM': SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
            'Polynomial SVM': SVC(kernel='poly', degree=3, C=1, random_state=42),
            'Sigmoid SVM': SVC(kernel='sigmoid', C=1, gamma='scale', random_state=42)
        }
        
        plt.figure(figsize=(20, 15))
        
        for i, (name, model) in enumerate(models_2d.items()):
            model.fit(X_train_pca, self.y_train)
            
            plt.subplot(2, 2, i + 1)
            
            # Create a mesh for decision boundary
            h = 0.1
            x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
            y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Get decision boundary
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            
            # Plot data points
            scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                                c=self.y_train, cmap='viridis', alpha=0.7, edgecolors='black')
            
            # Plot support vectors if available
            if hasattr(model, 'support_vectors_'):
                support_vectors = model.support_vectors_
                plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                          s=100, facecolors='none', edgecolors='red', linewidth=2)
            
            plt.xlabel(f'First Principal Component')
            plt.ylabel(f'Second Principal Component')
            plt.title(f'{name} - Decision Boundary')
            
            # Add colorbar
            if i == 1:  # Add colorbar once
                plt.colorbar(scatter, label='Diagnosis (0=Benign, 1=Malignant)')
        
        plt.tight_layout()
        plt.show()
        
        print("Note: Decision boundaries are shown in 2D PCA space.")
        print("Red circles indicate support vectors.")
        
    def feature_importance_analysis(self):
        """Analyze feature importance for linear SVM"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if self.linear_svm is not None:
            # Get feature weights from linear SVM
            feature_names = [col for col in self.df.columns if col not in ['id', 'diagnosis']]
            weights = np.abs(self.linear_svm.coef_[0])
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': weights
            }).sort_values('importance', ascending=False)
            
            print("Top 15 most important features (Linear SVM):")
            print(feature_importance.head(15))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            
            plt.subplot(1, 2, 1)
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Absolute Weight')
            plt.title('Top 15 Feature Importance (Linear SVM)')
            plt.gca().invert_yaxis()
            
            plt.subplot(1, 2, 2)
            # Plot all feature weights
            plt.bar(range(len(weights)), weights)
            plt.xlabel('Feature Index')
            plt.ylabel('Absolute Weight')
            plt.title('All Feature Weights (Linear SVM)')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("Linear SVM not trained yet!")
    
    def run_complete_analysis(self):
        """Run the complete SVM analysis pipeline"""
        print("STARTING COMPLETE SVM ANALYSIS")
        print("="*60)
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Visualize data
        self.visualize_data()
        
        # Train linear SVM
        self.train_linear_svm()
        
        # Train non-linear SVMs
        self.train_nonlinear_svms()
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Visualize results
        self.visualize_results(results)
        
        # Visualize decision boundaries
        self.visualize_decision_boundaries()
        
        # Feature importance analysis
        self.feature_importance_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"Best performing model: {best_model}")
        print(f"Best accuracy: {results[best_model]['accuracy']:.4f}")
        
        print(f"\nModel Performance Summary:")
        for model_name, metrics in results.items():
            print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}")


def main():
    """Main function to run the SVM analysis"""
    print("Support Vector Machines (SVM) Analysis")
    print("Dataset: Breast Cancer Wisconsin")
    print("="*60)
    
    # Initialize classifier
    svm_classifier = SVMClassifier('breast-cancer.csv')
    
    # Run complete analysis
    svm_classifier.run_complete_analysis()
    
    print(f"\nAnalysis completed!")
    print(f"This analysis demonstrated:")
    print(f"1. Linear vs Non-linear SVM classification")
    print(f"2. Different kernel functions (Linear, RBF, Polynomial, Sigmoid)")
    print(f"3. Hyperparameter tuning using GridSearchCV")
    print(f"4. Model evaluation and comparison")
    print(f"5. Decision boundary visualization")
    print(f"6. Feature importance analysis")


if __name__ == "__main__":
    main()
