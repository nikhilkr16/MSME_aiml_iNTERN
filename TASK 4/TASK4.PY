import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the breast cancer dataset"""
    print("=" * 60)
    print("TASK 4: BINARY CLASSIFICATION WITH LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('data.csv')
    
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    
    # Display basic statistics
    print(f"\nTarget variable distribution:")
    print(df['diagnosis'].value_counts())
    print(f"Malignant (M): {(df['diagnosis'] == 'M').sum()}")
    print(f"Benign (B): {(df['diagnosis'] == 'B').sum()}")
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\n2. DATA PREPROCESSING")
    print("-" * 30)
    
    # Drop the ID column and any unnamed columns
    columns_to_drop = []
    if 'id' in df.columns:
        columns_to_drop.append('id')
    
    # Drop any unnamed columns that contain NaN values
    for col in df.columns:
        if 'Unnamed' in col or df[col].isnull().all():
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
        print(f"Dropped columns: {columns_to_drop}")
    
    # Check for any remaining NaN values
    if df.isnull().sum().sum() > 0:
        print("Warning: Found NaN values, dropping rows with missing data")
        df = df.dropna()
    
    # Convert diagnosis to binary (1 for Malignant, 0 for Benign)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(X.columns[:5])}... (showing first 5)")
    
    return X, y

def create_visualizations(X, y):
    """Create exploratory data analysis visualizations"""
    print("\n3. EXPLORATORY DATA ANALYSIS")
    print("-" * 30)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Breast Cancer Dataset - Exploratory Data Analysis', fontsize=16)
    
    # Plot 1: Target distribution
    y_labels = ['Benign', 'Malignant']
    y_counts = [sum(y == 0), sum(y == 1)]
    axes[0, 0].pie(y_counts, labels=y_labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Distribution of Diagnosis')
    
    # Plot 2: Feature correlation heatmap (first 10 features)
    corr_matrix = X.iloc[:, :10].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0, 1], cbar_kws={'shrink': 0.8})
    axes[0, 1].set_title('Feature Correlation (First 10 Features)')
    
    # Plot 3: Box plot of key features
    key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    df_plot = pd.DataFrame(X[key_features])
    df_plot['diagnosis'] = y.map({0: 'Benign', 1: 'Malignant'})
    
    df_melted = df_plot.melt(id_vars=['diagnosis'], var_name='Feature', value_name='Value')
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='diagnosis', ax=axes[1, 0])
    axes[1, 0].set_title('Key Features by Diagnosis')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Feature importance preview (correlation with target)
    feature_corr = X.corrwith(y).abs().sort_values(ascending=False).head(10)
    axes[1, 1].barh(range(len(feature_corr)), feature_corr.values)
    axes[1, 1].set_yticks(range(len(feature_corr)))
    axes[1, 1].set_yticklabels(feature_corr.index)
    axes[1, 1].set_title('Top 10 Features by Correlation with Target')
    axes[1, 1].set_xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_logistic_regression(X, y):
    """Train and evaluate logistic regression model"""
    print("\n4. MODEL TRAINING AND EVALUATION")
    print("-" * 30)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba

def create_evaluation_plots(y_test, y_pred, y_pred_proba):
    """Create model evaluation visualizations"""
    print("\n5. MODEL EVALUATION VISUALIZATIONS")
    print("-" * 30)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Logistic Regression Model Evaluation', fontsize=16)
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xticklabels(['Benign', 'Malignant'])
    axes[0, 0].set_yticklabels(['Benign', 'Malignant'])
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # Plot 3: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    axes[1, 0].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True)
    
    # Plot 4: Prediction Probability Distribution
    benign_probs = y_pred_proba[y_test == 0]
    malignant_probs = y_pred_proba[y_test == 1]
    
    axes[1, 1].hist(benign_probs, bins=20, alpha=0.7, label='Benign', color='green')
    axes[1, 1].hist(malignant_probs, bins=20, alpha=0.7, label='Malignant', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Predicted Probabilities')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    print("\n6. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 30)
    
    # Get feature coefficients
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    
    colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
    bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Feature Coefficients in Logistic Regression')
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Positive (→ Malignant)'),
                      Patch(facecolor='red', label='Negative (→ Benign)')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def make_predictions_example(model, scaler, feature_names):
    """Demonstrate making predictions on new data"""
    print("\n7. PREDICTION EXAMPLE")
    print("-" * 30)
    
    # Load and preprocess the data in the same way as training
    df = pd.read_csv('data.csv')
    
    # Drop the same columns as during preprocessing
    columns_to_drop = []
    if 'id' in df.columns:
        columns_to_drop.append('id')
    
    for col in df.columns:
        if 'Unnamed' in col or df[col].isnull().all():
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
    
    # Get the feature columns only
    X_example = df.drop(['diagnosis'], axis=1)
    
    # Use a couple of real examples from the dataset
    sample_indices = [0, 100]  # First malignant and first benign cases
    
    for idx in sample_indices:
        sample = X_example.iloc[idx:idx+1]
        sample_scaled = scaler.transform(sample)
        
        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0]
        
        actual_diagnosis = df.iloc[idx]['diagnosis']
        
        print(f"\nSample {idx + 1}:")
        print(f"Actual diagnosis: {'Malignant' if actual_diagnosis == 'M' else 'Benign'}")
        print(f"Predicted diagnosis: {'Malignant' if prediction == 1 else 'Benign'}")
        print(f"Prediction probabilities: Benign: {probability[0]:.3f}, Malignant: {probability[1]:.3f}")

def main():
    """Main function to run the complete analysis"""
    try:
        # Load and explore data
        df = load_and_explore_data()
        
        # Preprocess data
        X, y = preprocess_data(df)
        
        # Create EDA visualizations
        create_visualizations(X, y)
        
        # Train model
        model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba = train_logistic_regression(X, y)
        
        # Create evaluation plots
        create_evaluation_plots(y_test, y_pred, y_pred_proba)
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(model, X.columns)
        
        # Make prediction examples
        make_predictions_example(model, scaler, X.columns)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("Generated files:")
        print("- eda_analysis.png: Exploratory Data Analysis")
        print("- model_evaluation.png: Model Performance Evaluation")
        print("- feature_importance.png: Feature Importance Analysis")
        print("=" * 60)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure the data.csv file is in the same directory.")

if __name__ == "__main__":
    main()
