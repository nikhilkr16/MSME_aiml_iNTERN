import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the Iris dataset and perform initial exploration"""
    print("=" * 60)
    print("IRIS DATASET - K-NEAREST NEIGHBORS CLASSIFICATION")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('Iris.csv')
    
    print("\n1. DATASET OVERVIEW:")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[1:-1])}")  # Exclude Id and Species
    print(f"Target classes: {df['Species'].unique()}")
    print(f"Class distribution:\n{df['Species'].value_counts()}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    
    return df

def visualize_data(df):
    """Create visualizations to understand the data"""
    print("\n2. DATA VISUALIZATION:")
    print("-" * 30)
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # Features for analysis (excluding Id and Species)
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # 1. Pairplot of the main distinguishing features
    ax1 = plt.subplot(2, 2, 1)
    for species in df['Species'].unique():
        species_data = df[df['Species'] == species]
        plt.scatter(species_data['PetalLengthCm'], species_data['PetalWidthCm'], 
                   label=species, alpha=0.7, s=60)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Petal Length vs Petal Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sepal comparison
    ax2 = plt.subplot(2, 2, 2)
    for species in df['Species'].unique():
        species_data = df[df['Species'] == species]
        plt.scatter(species_data['SepalLengthCm'], species_data['SepalWidthCm'], 
                   label=species, alpha=0.7, s=60)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Sepal Length vs Sepal Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot for feature distributions
    ax3 = plt.subplot(2, 2, 3)
    df_melted = df.melt(id_vars=['Species'], value_vars=features, 
                       var_name='Feature', value_name='Value')
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Species', ax=ax3)
    plt.title('Feature Distributions by Species')
    plt.xticks(rotation=45)
    
    # 4. Correlation heatmap
    ax4 = plt.subplot(2, 2, 4)
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax4)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

def prepare_data(df):
    """Prepare the data for machine learning"""
    print("\n3. DATA PREPARATION:")
    print("-" * 30)
    
    # Features and target
    X = df.drop(['Id', 'Species'], axis=1)
    y = df['Species']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features scaled using StandardScaler")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def find_optimal_k(X_train_scaled, y_train):
    """Find the optimal number of neighbors using cross-validation"""
    print("\n4. FINDING OPTIMAL K:")
    print("-" * 30)
    
    # Test different values of k
    k_range = range(1, 21)
    cv_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Find optimal k
    optimal_k = k_range[np.argmax(cv_scores)]
    best_score = max(cv_scores)
    
    print(f"Optimal k: {optimal_k}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN: Accuracy vs Number of Neighbors')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return optimal_k

def train_and_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, optimal_k):
    """Train KNN model and evaluate performance"""
    print("\n5. MODEL TRAINING AND EVALUATION:")
    print("-" * 30)
    
    # Train the KNN model with optimal k
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 5))
    
    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=knn.classes_, yticklabels=knn.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot prediction probabilities for test set
    plt.subplot(1, 2, 2)
    proba = knn.predict_proba(X_test_scaled)
    max_proba = np.max(proba, axis=1)
    colors = ['red' if pred != actual else 'green' 
              for pred, actual in zip(y_pred, y_test)]
    
    plt.scatter(range(len(max_proba)), max_proba, c=colors, alpha=0.6)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Maximum Prediction Probability')
    plt.title('Prediction Confidence\n(Green: Correct, Red: Incorrect)')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return knn

def visualize_decision_boundary(X_train_scaled, y_train, knn):
    """Visualize decision boundaries using PCA for 2D representation"""
    print("\n6. DECISION BOUNDARY VISUALIZATION:")
    print("-" * 30)
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    # Train KNN on PCA-transformed data
    knn_pca = KNeighborsClassifier(n_neighbors=knn.n_neighbors)
    knn_pca.fit(X_pca, y_train)
    
    # Create a mesh for plotting decision boundary
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # Plot training points
    species_list = np.unique(y_train)
    colors = ['red', 'blue', 'green']
    for i, species in enumerate(species_list):
        mask = y_train == species
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=species, s=60, alpha=0.8)
    
    plt.xlabel(f'First Principal Component\n(Explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Second Principal Component\n(Explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('KNN Decision Boundary (PCA Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PCA component analysis
    plt.subplot(1, 2, 2)
    feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    
    # Plot PCA components
    x_pos = np.arange(len(feature_names))
    plt.bar(x_pos - 0.2, pca.components_[0], 0.4, label='PC1', alpha=0.7)
    plt.bar(x_pos + 0.2, pca.components_[1], 0.4, label='PC2', alpha=0.7)
    
    plt.xlabel('Features')
    plt.ylabel('Component Weight')
    plt.title('PCA Components Analysis')
    plt.xticks(x_pos, feature_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_):.2%}")

def hyperparameter_tuning(X_train_scaled, y_train):
    """Perform hyperparameter tuning using GridSearchCV"""
    print("\n7. HYPERPARAMETER TUNING:")
    print("-" * 30)
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # GridSearchCV
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    """Main function to run the complete KNN analysis"""
    # Load and explore data
    df = load_and_explore_data()
    
    # Visualize data
    visualize_data(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)
    
    # Find optimal k
    optimal_k = find_optimal_k(X_train_scaled, y_train)
    
    # Train and evaluate model
    knn = train_and_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, optimal_k)
    
    # Visualize decision boundary
    visualize_decision_boundary(X_train_scaled, y_train, knn)
    
    # Hyperparameter tuning
    best_knn = hyperparameter_tuning(X_train_scaled, y_train)
    
    # Final evaluation with best model
    print("\n8. FINAL MODEL PERFORMANCE:")
    print("-" * 30)
    best_knn.fit(X_train_scaled, y_train)
    final_accuracy = best_knn.score(X_test_scaled, y_test)
    print(f"Final model accuracy: {final_accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("KNN CLASSIFICATION ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
