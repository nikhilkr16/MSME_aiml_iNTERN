"""
Decision Trees and Random Forests for Regression
Supplementary example demonstrating regression capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

def create_regression_dataset():
    """Create a synthetic regression dataset"""
    print("=" * 60)
    print("CREATING SYNTHETIC REGRESSION DATASET")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, 
                          noise=0.1, random_state=42)
    
    # Add some non-linear relationships
    X[:, 0] = X[:, 0] ** 2  # Quadratic relationship
    X[:, 1] = np.sin(X[:, 1])  # Sinusoidal relationship
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target statistics:")
    print(df['target'].describe())
    
    return df, X, y, feature_names

def visualize_regression_data(df, X, y):
    """Visualize the regression dataset"""
    print("\n" + "=" * 60)
    print("REGRESSION DATA VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Regression Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Target distribution
    axes[0, 0].hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].set_xlabel('Target Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Feature vs Target scatter plots
    for i in range(4):
        row = i // 2
        col = (i % 2) + 1
        if col < 3:
            axes[row, col].scatter(X[:, i], y, alpha=0.6, s=10)
            axes[row, col].set_title(f'Feature {i} vs Target')
            axes[row, col].set_xlabel(f'Feature {i}')
            axes[row, col].set_ylabel('Target')
    
    # 5. Correlation matrix (subset of features)
    subset_df = df[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'target']]
    corr_matrix = subset_df.corr()
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45)
    axes[1, 2].set_yticklabels(corr_matrix.columns)
    axes[1, 2].set_title('Feature Correlation Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()

def train_regression_models(X, y, feature_names):
    """Train Decision Tree and Random Forest regressors"""
    print("\n" + "=" * 60)
    print("TRAINING REGRESSION MODELS")
    print("=" * 60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Train Decision Tree Regressor
    print("\n" + "-" * 40)
    print("DECISION TREE REGRESSOR")
    print("-" * 40)
    
    dt_reg = DecisionTreeRegressor(random_state=42, max_depth=10)
    dt_reg.fit(X_train, y_train)
    
    dt_train_pred = dt_reg.predict(X_train)
    dt_test_pred = dt_reg.predict(X_test)
    
    dt_train_mse = mean_squared_error(y_train, dt_train_pred)
    dt_test_mse = mean_squared_error(y_test, dt_test_pred)
    dt_train_r2 = r2_score(y_train, dt_train_pred)
    dt_test_r2 = r2_score(y_test, dt_test_pred)
    
    print(f"Training MSE: {dt_train_mse:.4f}")
    print(f"Test MSE: {dt_test_mse:.4f}")
    print(f"Training R²: {dt_train_r2:.4f}")
    print(f"Test R²: {dt_test_r2:.4f}")
    
    # Train Random Forest Regressor
    print("\n" + "-" * 40)
    print("RANDOM FOREST REGRESSOR")
    print("-" * 40)
    
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_reg.fit(X_train, y_train)
    
    rf_train_pred = rf_reg.predict(X_train)
    rf_test_pred = rf_reg.predict(X_test)
    
    rf_train_mse = mean_squared_error(y_train, rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, rf_test_pred)
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    
    print(f"Training MSE: {rf_train_mse:.4f}")
    print(f"Test MSE: {rf_test_mse:.4f}")
    print(f"Training R²: {rf_train_r2:.4f}")
    print(f"Test R²: {rf_test_r2:.4f}")
    
    # Feature importance
    print("\n" + "-" * 40)
    print("FEATURE IMPORTANCE COMPARISON")
    print("-" * 40)
    
    dt_importance = pd.DataFrame({
        'feature': feature_names,
        'dt_importance': dt_reg.feature_importances_,
        'rf_importance': rf_reg.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    print(dt_importance)
    
    return (dt_reg, rf_reg, X_train, X_test, y_train, y_test, 
            dt_test_pred, rf_test_pred, dt_importance)

def visualize_regression_results(dt_reg, rf_reg, X_train, X_test, y_train, y_test, 
                               dt_test_pred, rf_test_pred, dt_importance, feature_names):
    """Visualize regression results"""
    print("\n" + "=" * 60)
    print("REGRESSION RESULTS VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Regression Models Comparison', fontsize=16, fontweight='bold')
    
    # 1. Decision Tree predictions vs actual
    axes[0, 0].scatter(y_test, dt_test_pred, alpha=0.6, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Decision Tree: Predictions vs Actual')
    dt_r2 = r2_score(y_test, dt_test_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {dt_r2:.3f}', transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Random Forest predictions vs actual
    axes[0, 1].scatter(y_test, rf_test_pred, alpha=0.6, s=20, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Random Forest: Predictions vs Actual')
    rf_r2 = r2_score(y_test, rf_test_pred)
    axes[0, 1].text(0.05, 0.95, f'R² = {rf_r2:.3f}', transform=axes[0, 1].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Residuals plot
    dt_residuals = y_test - dt_test_pred
    rf_residuals = y_test - rf_test_pred
    
    axes[0, 2].scatter(dt_test_pred, dt_residuals, alpha=0.6, s=20, label='Decision Tree')
    axes[0, 2].scatter(rf_test_pred, rf_residuals, alpha=0.6, s=20, label='Random Forest')
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Predicted Values')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residuals Plot')
    axes[0, 2].legend()
    
    # 4. Feature importance comparison
    x_pos = np.arange(len(dt_importance))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, dt_importance['dt_importance'], width, 
                   label='Decision Tree', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2, dt_importance['rf_importance'], width, 
                   label='Random Forest', alpha=0.8)
    axes[1, 0].set_xlabel('Features')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].set_title('Feature Importance Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(dt_importance['feature'], rotation=45)
    axes[1, 0].legend()
    
    # 5. Model performance comparison
    models = ['Decision Tree', 'Random Forest']
    mse_scores = [mean_squared_error(y_test, dt_test_pred), 
                  mean_squared_error(y_test, rf_test_pred)]
    r2_scores = [r2_score(y_test, dt_test_pred), 
                 r2_score(y_test, rf_test_pred)]
    
    x = np.arange(len(models))
    axes[1, 1].bar(x, mse_scores, alpha=0.7, color=['skyblue', 'lightgreen'])
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Mean Squared Error')
    axes[1, 1].set_title('MSE Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    
    # Add values on bars
    for i, v in enumerate(mse_scores):
        axes[1, 1].text(i, v + max(mse_scores) * 0.01, f'{v:.3f}', 
                        ha='center', va='bottom')
    
    # 6. Learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    dt_train_scores = []
    dt_val_scores = []
    rf_train_scores = []
    rf_val_scores = []
    
    for train_size in train_sizes:
        n_samples = int(train_size * len(X_train))
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        # Decision Tree
        dt_temp = DecisionTreeRegressor(random_state=42, max_depth=10)
        dt_temp.fit(X_subset, y_subset)
        dt_train_scores.append(r2_score(y_subset, dt_temp.predict(X_subset)))
        dt_val_scores.append(r2_score(y_test, dt_temp.predict(X_test)))
        
        # Random Forest
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_temp.fit(X_subset, y_subset)
        rf_train_scores.append(r2_score(y_subset, rf_temp.predict(X_subset)))
        rf_val_scores.append(r2_score(y_test, rf_temp.predict(X_test)))
    
    axes[1, 2].plot(train_sizes, dt_train_scores, 'o-', label='DT Train')
    axes[1, 2].plot(train_sizes, dt_val_scores, 's-', label='DT Test')
    axes[1, 2].plot(train_sizes, rf_train_scores, '^-', label='RF Train')
    axes[1, 2].plot(train_sizes, rf_val_scores, 'd-', label='RF Test')
    axes[1, 2].set_xlabel('Training Set Size')
    axes[1, 2].set_ylabel('R² Score')
    axes[1, 2].set_title('Learning Curves')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_decision_tree_regression(dt_reg, feature_names):
    """Visualize the decision tree for regression"""
    print("\n" + "=" * 60)
    print("DECISION TREE VISUALIZATION (REGRESSION)")
    print("=" * 60)
    
    # Create a simplified tree for visualization
    simple_dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    
    # Use a subset of training data for cleaner visualization
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=200, replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    simple_dt.fit(X_sample, y_sample)
    
    # Plot the tree
    plt.figure(figsize=(20, 12))
    plot_tree(simple_dt, 
              feature_names=feature_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Decision Tree for Regression (max_depth=3)', fontsize=16, fontweight='bold')
    plt.show()
    
    print(f"Tree depth: {simple_dt.get_depth()}")
    print(f"Number of leaves: {simple_dt.get_n_leaves()}")

def demonstrate_overfitting():
    """Demonstrate overfitting in decision trees"""
    print("\n" + "=" * 60)
    print("OVERFITTING DEMONSTRATION")
    print("=" * 60)
    
    # Create simple dataset
    np.random.seed(42)
    X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
    y_simple = np.sin(X_simple.flatten()) + np.random.normal(0, 0.1, 100)
    
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y_simple, test_size=0.3, random_state=42
    )
    
    # Train trees with different depths
    max_depths = [1, 3, 5, 10, None]
    
    fig, axes = plt.subplots(1, len(max_depths), figsize=(20, 4))
    fig.suptitle('Decision Tree Overfitting with Increasing Depth', fontsize=16, fontweight='bold')
    
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    
    for i, max_depth in enumerate(max_depths):
        # Train model
        dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        dt.fit(X_train_simple, y_train_simple)
        
        # Predict
        y_plot = dt.predict(X_plot)
        train_score = r2_score(y_train_simple, dt.predict(X_train_simple))
        test_score = r2_score(y_test_simple, dt.predict(X_test_simple))
        
        # Plot
        axes[i].scatter(X_train_simple, y_train_simple, alpha=0.6, s=20, label='Training')
        axes[i].scatter(X_test_simple, y_test_simple, alpha=0.6, s=20, label='Test')
        axes[i].plot(X_plot, y_plot, color='red', linewidth=2, label='Prediction')
        axes[i].set_title(f'Max Depth: {max_depth}\nTrain R²: {train_score:.3f}\nTest R²: {test_score:.3f}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('y')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for regression demonstration"""
    print("=" * 80)
    print("DECISION TREES AND RANDOM FORESTS FOR REGRESSION")
    print("=" * 80)
    
    # Create dataset
    df, X, y, feature_names = create_regression_dataset()
    
    # Visualize data
    visualize_regression_data(df, X, y)
    
    # Train models
    results = train_regression_models(X, y, feature_names)
    dt_reg, rf_reg, X_train, X_test, y_train, y_test, dt_test_pred, rf_test_pred, dt_importance = results
    
    # Visualize results
    visualize_regression_results(dt_reg, rf_reg, X_train, X_test, y_train, y_test, 
                               dt_test_pred, rf_test_pred, dt_importance, feature_names)
    
    # Visualize decision tree
    visualize_decision_tree_regression(dt_reg, feature_names)
    
    # Demonstrate overfitting
    demonstrate_overfitting()
    
    print("\n" + "=" * 80)
    print("REGRESSION ANALYSIS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main() 