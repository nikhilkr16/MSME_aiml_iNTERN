"""
TASK 5: Decision Trees and Random Forests
Objective: Learn tree-based models for classification & regression
Dataset: Heart Disease Prediction
Tools: Scikit-learn, Graphviz, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HeartDiseaseAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dt_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        
    def load_and_explore_data(self):
        """Load and explore the heart disease dataset"""
        print("=" * 60)
        print("LOADING AND EXPLORING HEART DISEASE DATASET")
        print("=" * 60)
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nDataset info:")
        print(self.df.info())
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        print(f"\nTarget variable distribution:")
        print(self.df['target'].value_counts())
        
        print(f"\nBasic statistics:")
        print(self.df.describe())
        
        # Feature descriptions
        feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of the peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (1-3)',
            'target': 'Heart disease (1 = presence; 0 = absence)'
        }
        
        print(f"\nFeature Descriptions:")
        for feature, description in feature_descriptions.items():
            print(f"{feature}: {description}")
        
        return self.df
    
    def visualize_data(self):
        """Create visualizations for data exploration"""
        print("\n" + "=" * 60)
        print("DATA VISUALIZATION")
        print("=" * 60)
        
        # Set up the plotting area
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Heart Disease Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        axes[0, 0].pie(self.df['target'].value_counts(), 
                       labels=['No Disease', 'Disease'], 
                       autopct='%1.1f%%', 
                       colors=['lightcoral', 'lightblue'])
        axes[0, 0].set_title('Target Distribution')
        
        # 2. Age distribution by target
        self.df.boxplot(column='age', by='target', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Target')
        axes[0, 1].set_xlabel('Target')
        axes[0, 1].set_ylabel('Age')
        
        # 3. Correlation heatmap
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    ax=axes[1, 0], fmt='.2f', square=True)
        axes[1, 0].set_title('Feature Correlation Matrix')
        
        # 4. Feature importance by correlation with target
        target_corr = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
        axes[1, 1].barh(range(len(target_corr)), target_corr.values)
        axes[1, 1].set_yticks(range(len(target_corr)))
        axes[1, 1].set_yticklabels(target_corr.index)
        axes[1, 1].set_title('Feature Correlation with Target')
        axes[1, 1].set_xlabel('Correlation')
        
        # 5. Chest pain type distribution
        cp_counts = self.df.groupby(['cp', 'target']).size().unstack()
        cp_counts.plot(kind='bar', ax=axes[2, 0], color=['lightcoral', 'lightblue'])
        axes[2, 0].set_title('Chest Pain Type vs Target')
        axes[2, 0].set_xlabel('Chest Pain Type')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].legend(['No Disease', 'Disease'])
        axes[2, 0].tick_params(axis='x', rotation=0)
        
        # 6. Max heart rate vs age colored by target
        colors = ['red' if x == 1 else 'blue' for x in self.df['target']]
        scatter = axes[2, 1].scatter(self.df['age'], self.df['thalach'], 
                                    c=colors, alpha=0.6)
        axes[2, 1].set_title('Max Heart Rate vs Age')
        axes[2, 1].set_xlabel('Age')
        axes[2, 1].set_ylabel('Max Heart Rate')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                 markersize=10, label='No Disease'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                 markersize=10, label='Disease')]
        axes[2, 1].legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for modeling"""
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        
        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale the features (important for some tree-based visualizations)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preparation completed!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self):
        """Train and evaluate Decision Tree model"""
        print("\n" + "=" * 60)
        print("DECISION TREE CLASSIFIER")
        print("=" * 60)
        
        # Train decision tree with default parameters
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.dt_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        dt_train_pred = self.dt_model.predict(self.X_train)
        dt_test_pred = self.dt_model.predict(self.X_test)
        dt_test_proba = self.dt_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate model
        train_accuracy = accuracy_score(self.y_train, dt_train_pred)
        test_accuracy = accuracy_score(self.y_test, dt_test_pred)
        auc_score = roc_auc_score(self.y_test, dt_test_proba)
        
        print(f"Decision Tree Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.dt_model, self.X_train, self.y_train, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, dt_test_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances:")
        print(feature_importance.head(10))
        
        return self.dt_model, feature_importance
    
    def optimize_decision_tree(self):
        """Optimize Decision Tree hyperparameters"""
        print("\n" + "=" * 40)
        print("DECISION TREE HYPERPARAMETER OPTIMIZATION")
        print("=" * 40)
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        # Perform grid search
        dt_grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        dt_grid.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {dt_grid.best_params_}")
        print(f"Best cross-validation score: {dt_grid.best_score_:.4f}")
        
        # Train optimized model
        self.dt_optimized = dt_grid.best_estimator_
        dt_opt_pred = self.dt_optimized.predict(self.X_test)
        dt_opt_accuracy = accuracy_score(self.y_test, dt_opt_pred)
        
        print(f"Optimized Decision Tree Test Accuracy: {dt_opt_accuracy:.4f}")
        
        return self.dt_optimized
    
    def train_random_forest(self):
        """Train and evaluate Random Forest model"""
        print("\n" + "=" * 60)
        print("RANDOM FOREST CLASSIFIER")
        print("=" * 60)
        
        # Train random forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        rf_train_pred = self.rf_model.predict(self.X_train)
        rf_test_pred = self.rf_model.predict(self.X_test)
        rf_test_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate model
        train_accuracy = accuracy_score(self.y_train, rf_train_pred)
        test_accuracy = accuracy_score(self.y_test, rf_test_pred)
        auc_score = roc_auc_score(self.y_test, rf_test_proba)
        
        print(f"Random Forest Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, rf_test_pred))
        
        # Feature importance
        rf_feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances:")
        print(rf_feature_importance.head(10))
        
        return self.rf_model, rf_feature_importance
    
    def optimize_random_forest(self):
        """Optimize Random Forest hyperparameters"""
        print("\n" + "=" * 40)
        print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
        print("=" * 40)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        # Perform grid search
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,  # Reduced CV for faster execution
            scoring='accuracy',
            n_jobs=-1
        )
        
        rf_grid.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"Best cross-validation score: {rf_grid.best_score_:.4f}")
        
        # Train optimized model
        self.rf_optimized = rf_grid.best_estimator_
        rf_opt_pred = self.rf_optimized.predict(self.X_test)
        rf_opt_accuracy = accuracy_score(self.y_test, rf_opt_pred)
        
        print(f"Optimized Random Forest Test Accuracy: {rf_opt_accuracy:.4f}")
        
        return self.rf_optimized
    
    def visualize_tree(self, max_depth=3):
        """Visualize decision tree"""
        print("\n" + "=" * 60)
        print("DECISION TREE VISUALIZATION")
        print("=" * 60)
        
        # Create a simplified tree for visualization
        simple_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        simple_dt.fit(self.X_train, self.y_train)
        
        # Plot the tree
        plt.figure(figsize=(20, 12))
        plot_tree(simple_dt, 
                  feature_names=self.X_train.columns,
                  class_names=['No Disease', 'Disease'],
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(f'Decision Tree Visualization (max_depth={max_depth})', fontsize=16, fontweight='bold')
        plt.show()
        
        # Print text representation
        print(f"\nText representation of the tree (max_depth={max_depth}):")
        tree_rules = export_text(simple_dt, feature_names=list(self.X_train.columns))
        print(tree_rules[:2000] + "..." if len(tree_rules) > 2000 else tree_rules)
    
    def compare_models(self):
        """Compare all models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        models = {
            'Decision Tree': self.dt_model,
            'Optimized Decision Tree': self.dt_optimized,
            'Random Forest': self.rf_model,
            'Optimized Random Forest': self.rf_optimized
        }
        
        results = []
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'AUC': auc
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Accuracy comparison
        axes[0].bar(results_df['Model'], results_df['Accuracy'])
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. ROC curves
        for name, model in models.items():
            y_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            axes[1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curves')
        axes[1].legend()
        
        # 3. Feature importance comparison
        dt_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        rf_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        x = np.arange(len(dt_importance))
        width = 0.35
        
        axes[2].bar(x - width/2, dt_importance['importance'], width, label='Decision Tree')
        axes[2].bar(x + width/2, rf_importance['importance'], width, label='Random Forest')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Importance')
        axes[2].set_title('Top 10 Feature Importances')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(dt_importance['feature'], rotation=45)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def analyze_model_insights(self):
        """Provide insights about the models"""
        print("\n" + "=" * 60)
        print("MODEL INSIGHTS AND INTERPRETATION")
        print("=" * 60)
        
        print("Key Findings:")
        print("-" * 40)
        
        # Feature importance insights
        rf_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"1. Most Important Features for Heart Disease Prediction:")
        for i, (_, row) in enumerate(rf_importance.head(5).iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        print(f"\n2. Model Performance Summary:")
        dt_accuracy = accuracy_score(self.y_test, self.dt_model.predict(self.X_test))
        rf_accuracy = accuracy_score(self.y_test, self.rf_model.predict(self.X_test))
        
        print(f"   - Decision Tree Accuracy: {dt_accuracy:.4f}")
        print(f"   - Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"   - Random Forest shows {'better' if rf_accuracy > dt_accuracy else 'similar'} performance")
        
        print(f"\n3. Model Characteristics:")
        print(f"   - Decision Tree: High interpretability, prone to overfitting")
        print(f"   - Random Forest: Better generalization, ensemble approach")
        print(f"   - Random Forest uses {self.rf_model.n_estimators} decision trees")
        
        print(f"\n4. Clinical Insights:")
        top_features = rf_importance.head(3)['feature'].tolist()
        feature_meanings = {
            'cp': 'chest pain type',
            'thalach': 'maximum heart rate achieved',
            'oldpeak': 'ST depression induced by exercise',
            'ca': 'number of major vessels colored by fluoroscopy',
            'thal': 'thalassemia',
            'age': 'age',
            'sex': 'gender',
            'exang': 'exercise induced angina'
        }
        
        print(f"   The top predictive factors are:")
        for feature in top_features:
            meaning = feature_meanings.get(feature, feature)
            print(f"   - {feature} ({meaning})")

def main():
    """Main function to run the complete analysis"""
    print("=" * 80)
    print("HEART DISEASE PREDICTION USING DECISION TREES AND RANDOM FORESTS")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = HeartDiseaseAnalyzer('heart.csv')
    
    # Step 1: Load and explore data
    analyzer.load_and_explore_data()
    
    # Step 2: Visualize data
    analyzer.visualize_data()
    
    # Step 3: Prepare data
    analyzer.prepare_data()
    
    # Step 4: Train Decision Tree
    analyzer.train_decision_tree()
    
    # Step 5: Optimize Decision Tree
    analyzer.optimize_decision_tree()
    
    # Step 6: Train Random Forest
    analyzer.train_random_forest()
    
    # Step 7: Optimize Random Forest
    analyzer.optimize_random_forest()
    
    # Step 8: Visualize Decision Tree
    analyzer.visualize_tree(max_depth=3)
    
    # Step 9: Compare models
    analyzer.compare_models()
    
    # Step 10: Provide insights
    analyzer.analyze_model_insights()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
