"""
Simple Demo: Decision Trees and Random Forests
A quick demonstration of tree-based models for heart disease prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the heart disease dataset"""
    print("🔍 Loading Heart Disease Dataset...")
    
    # Load dataset
    df = pd.read_csv('heart.csv')
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"🎯 Target Distribution:")
    print(df['target'].value_counts())
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test, df

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train and evaluate Decision Tree"""
    print("\n🌳 Training Decision Tree...")
    
    # Train model
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📈 Decision Tree Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 5 Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
    
    return dt_model, feature_importance

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest"""
    print("\n🌲 Training Random Forest...")
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📈 Random Forest Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 5 Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
    
    return rf_model, feature_importance

def visualize_results(dt_model, rf_model, X_train, X_test, y_train, y_test, dt_importance, rf_importance):
    """Create visualizations"""
    print("\n📊 Creating Visualizations...")
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Decision Trees vs Random Forest Analysis', fontsize=16, fontweight='bold')
    
    # 1. Decision Tree visualization (simplified)
    simple_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    simple_dt.fit(X_train, y_train)
    
    plot_tree(simple_dt, 
              feature_names=X_train.columns,
              class_names=['No Disease', 'Disease'],
              filled=True,
              rounded=True,
              fontsize=8,
              ax=axes[0, 0])
    axes[0, 0].set_title('Decision Tree (max_depth=3)')
    
    # 2. Model Accuracy Comparison
    dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    
    models = ['Decision Tree', 'Random Forest']
    accuracies = [dt_accuracy, rf_accuracy]
    colors = ['skyblue', 'lightgreen']
    
    bars = axes[0, 1].bar(models, accuracies, color=colors, alpha=0.7)
    axes[0, 1].set_title('Model Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)
    
    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Feature Importance Comparison (Top 8 features)
    top_features = dt_importance.head(8)
    x = np.arange(len(top_features))
    width = 0.35
    
    # Get corresponding RF importance for same features
    rf_imp_dict = dict(zip(rf_importance['feature'], rf_importance['importance']))
    rf_imp_values = [rf_imp_dict.get(feat, 0) for feat in top_features['feature']]
    
    axes[1, 0].bar(x - width/2, top_features['importance'], width, 
                   label='Decision Tree', alpha=0.8, color='skyblue')
    axes[1, 0].bar(x + width/2, rf_imp_values, width, 
                   label='Random Forest', alpha=0.8, color='lightgreen')
    
    axes[1, 0].set_xlabel('Features')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].set_title('Feature Importance Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(top_features['feature'], rotation=45)
    axes[1, 0].legend()
    
    # 4. Confusion Matrix for Random Forest
    rf_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, rf_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=axes[1, 1])
    axes[1, 1].set_title('Random Forest Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def compare_models(dt_model, rf_model, X_test, y_test):
    """Compare model performance"""
    print("\n⚖️  Model Comparison:")
    print("=" * 50)
    
    # Get predictions
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    dt_accuracy = accuracy_score(y_test, dt_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    if rf_accuracy > dt_accuracy:
        print(f"🏆 Random Forest wins by {rf_accuracy - dt_accuracy:.4f}")
    elif dt_accuracy > rf_accuracy:
        print(f"🏆 Decision Tree wins by {dt_accuracy - rf_accuracy:.4f}")
    else:
        print(f"🤝 It's a tie!")
    
    print(f"\n📋 Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=['No Disease', 'Disease']))

def interpret_results(rf_importance):
    """Provide medical interpretation"""
    print("\n🏥 Medical Insights:")
    print("=" * 50)
    
    # Feature meanings
    feature_meanings = {
        'cp': 'Chest pain type - Different types indicate varying risk levels',
        'thalach': 'Maximum heart rate - Lower rates may indicate heart problems',
        'oldpeak': 'ST depression - ECG changes during exercise stress',
        'ca': 'Number of major vessels - More blocked vessels = higher risk',
        'thal': 'Thalassemia - Blood disorder affecting heart health',
        'age': 'Age - Risk increases with age',
        'sex': 'Gender - Men typically have higher risk',
        'exang': 'Exercise angina - Chest pain during exercise'
    }
    
    print("🔍 Top Risk Factors Identified:")
    for i, (_, row) in enumerate(rf_importance.head(5).iterrows(), 1):
        feature = row['feature']
        importance = row['importance']
        meaning = feature_meanings.get(feature, 'Medical feature')
        print(f"   {i}. {feature.upper()} ({importance:.3f})")
        print(f"      → {meaning}")

def main():
    """Main demo function"""
    print("🫀 HEART DISEASE PREDICTION DEMO")
    print("Using Decision Trees and Random Forests")
    print("=" * 60)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, df = load_and_prepare_data()
        
        # Train Decision Tree
        dt_model, dt_importance = train_decision_tree(X_train, X_test, y_train, y_test)
        
        # Train Random Forest
        rf_model, rf_importance = train_random_forest(X_train, X_test, y_train, y_test)
        
        # Compare models
        compare_models(dt_model, rf_model, X_test, y_test)
        
        # Create visualizations
        visualize_results(dt_model, rf_model, X_train, X_test, y_train, y_test, 
                         dt_importance, rf_importance)
        
        # Interpret results
        interpret_results(rf_importance)
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • Random Forest typically outperforms single Decision Tree")
        print("   • Chest pain type is the strongest predictor")
        print("   • Tree models provide excellent interpretability")
        print("   • Feature importance helps understand medical risk factors")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn")

if __name__ == "__main__":
    main() 