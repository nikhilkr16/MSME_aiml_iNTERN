import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("HOUSING PRICE PREDICTION - LINEAR REGRESSION")
    print("=" * 60)
    
    # 1. Load and explore the data
    print("\n📊 STEP 1: LOADING DATA")
    print("-" * 30)
    
    try:
        df = pd.read_csv('archive (1)/Housing.csv')
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {list(df.columns)}")
        
        print(f"\n📈 Price Statistics:")
        print(f"   Mean: ${df['price'].mean():,.0f}")
        print(f"   Min: ${df['price'].min():,.0f}")
        print(f"   Max: ${df['price'].max():,.0f}")
        
    except FileNotFoundError:
        print("❌ Error: Housing.csv file not found!")
        return
    
    # 2. Data preprocessing
    print(f"\n🔄 STEP 2: DATA PREPROCESSING")
    print("-" * 30)
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Convert categorical variables to numerical
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea']
    
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
    
    # Handle furnishing status
    furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    df_processed['furnishingstatus'] = df_processed['furnishingstatus'].map(furnishing_map)
    
    print("✅ Categorical variables converted to numerical")
    
    # Show top correlations with price
    correlations = df_processed.corr()['price'].sort_values(ascending=False)
    print(f"\n🔗 Top correlations with price:")
    for i, (feature, corr) in enumerate(correlations.head(6).items()):
        if feature != 'price':
            print(f"   {i}. {feature:15}: {corr:6.3f}")
    
    # 3. Simple Linear Regression (Price vs Area)
    print(f"\n🔸 STEP 3: SIMPLE LINEAR REGRESSION")
    print("-" * 30)
    
    X_simple = df_processed[['area']]
    y = df_processed['price']
    
    # Split data
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    
    # Train model
    simple_model = LinearRegression()
    simple_model.fit(X_train_simple, y_train_simple)
    
    # Make predictions
    y_pred_simple = simple_model.predict(X_test_simple)
    
    # Calculate metrics
    simple_r2 = r2_score(y_test_simple, y_pred_simple)
    simple_rmse = np.sqrt(mean_squared_error(y_test_simple, y_pred_simple))
    simple_mae = mean_absolute_error(y_test_simple, y_pred_simple)
    
    print(f"📐 Equation: Price = {simple_model.intercept_:.0f} + {simple_model.coef_[0]:.2f} × Area")
    print(f"📊 Performance:")
    print(f"   R² Score: {simple_r2:.4f} ({simple_r2*100:.1f}% variance explained)")
    print(f"   RMSE: ${simple_rmse:,.0f}")
    print(f"   MAE: ${simple_mae:,.0f}")
    
    # 4. Multiple Linear Regression
    print(f"\n🔹 STEP 4: MULTIPLE LINEAR REGRESSION")
    print("-" * 30)
    
    # Select all features except price
    feature_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                    'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                    'parking', 'prefarea', 'furnishingstatus']
    
    X_multiple = df_processed[feature_cols]
    
    # Split data
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multiple, y, test_size=0.2, random_state=42
    )
    
    # Train model
    multiple_model = LinearRegression()
    multiple_model.fit(X_train_multi, y_train_multi)
    
    # Make predictions
    y_pred_multi = multiple_model.predict(X_test_multi)
    
    # Calculate metrics
    multi_r2 = r2_score(y_test_multi, y_pred_multi)
    multi_rmse = np.sqrt(mean_squared_error(y_test_multi, y_pred_multi))
    multi_mae = mean_absolute_error(y_test_multi, y_pred_multi)
    
    print(f"📊 Performance:")
    print(f"   R² Score: {multi_r2:.4f} ({multi_r2*100:.1f}% variance explained)")
    print(f"   RMSE: ${multi_rmse:,.0f}")
    print(f"   MAE: ${multi_mae:,.0f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': multiple_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"\n🏆 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
        direction = "↗️" if row['Coefficient'] > 0 else "↘️"
        print(f"   {i+1}. {row['Feature']:15}: {row['Coefficient']:8.0f} {direction}")
    
    # 5. Model Comparison
    print(f"\n⚖️  STEP 5: MODEL COMPARISON")
    print("-" * 30)
    
    print(f"{'Metric':<12} {'Simple':<12} {'Multiple':<12} {'Improvement'}")
    print(f"{'-'*50}")
    print(f"{'R² Score':<12} {simple_r2:<12.4f} {multi_r2:<12.4f} {((multi_r2-simple_r2)/simple_r2*100):+.1f}%")
    print(f"{'RMSE ($)':<12} {simple_rmse:<12,.0f} {multi_rmse:<12,.0f} {((simple_rmse-multi_rmse)/simple_rmse*100):+.1f}%")
    print(f"{'MAE ($)':<12} {simple_mae:<12,.0f} {multi_mae:<12,.0f} {((simple_mae-multi_mae)/simple_mae*100):+.1f}%")
    
    # 6. Prediction Example
    print(f"\n🏠 STEP 6: SAMPLE PREDICTION")
    print("-" * 30)
    
    # Sample house
    sample_area = 7000
    sample_features = [7000, 3, 2, 2, 1, 0, 1, 0, 1, 2, 1, 1]  # All features in order
    
    simple_prediction = simple_model.predict([[sample_area]])[0]
    multi_prediction = multiple_model.predict([sample_features])[0]
    
    print(f"🏡 Sample House: {sample_area:,} sq ft, 3 bed, 2 bath, A/C, Preferred area")
    print(f"💰 Predictions:")
    print(f"   Simple Model: ${simple_prediction:,.0f}")
    print(f"   Multiple Model: ${multi_prediction:,.0f}")
    print(f"   Difference: ${abs(multi_prediction - simple_prediction):,.0f}")
    
    # 7. Quick Visualization
    print(f"\n📊 STEP 7: CREATING VISUALIZATION")
    print("-" * 30)
    
    plt.figure(figsize=(12, 4))
    
    # Simple regression plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_test_simple, y_test_simple, alpha=0.6, color='lightblue', s=30, label='Actual')
    plt.scatter(X_test_simple, y_pred_simple, alpha=0.6, color='red', s=30, label='Predicted')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.title(f'Simple Linear Regression\nR² = {simple_r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Multiple regression: Actual vs Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_multi, y_pred_multi, alpha=0.6, color='green', s=30)
    min_price, max_price = y_test_multi.min(), y_test_multi.max()
    plt.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Multiple Linear Regression\nR² = {multi_r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('housing_regression_analysis.png', dpi=100, bbox_inches='tight')
    print("✅ Visualization saved as 'housing_regression_analysis.png'")
    plt.show()
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 LINEAR REGRESSION ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\n📚 Key Findings:")
    print(f"   • Simple model (area only) explains {simple_r2*100:.1f}% of price variance")
    print(f"   • Multiple model (all features) explains {multi_r2*100:.1f}% of price variance")
    print(f"   • Adding more features improved accuracy by {((multi_r2-simple_r2)/simple_r2*100):.1f}%")
    print(f"   • Most important factor: {feature_importance.iloc[0]['Feature']}")
    print(f"   • Average prediction error: ${multi_mae:,.0f}")

if __name__ == "__main__":
    main() 