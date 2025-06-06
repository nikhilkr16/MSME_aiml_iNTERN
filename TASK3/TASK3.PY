import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousingLinearRegression:
    def __init__(self, data_path):
        """Initialize the class with data path"""
        self.data_path = data_path
        self.df = None
        self.simple_model = None
        self.multiple_model = None
        
    def load_and_explore_data(self):
        """Load and explore the housing dataset"""
        print("=" * 60)
        print("HOUSING PRICE PREDICTION - LINEAR REGRESSION")
        print("=" * 60)
        
        # Load the data
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n📊 Dataset Shape: {self.df.shape}")
        print(f"📋 Features: {list(self.df.columns)}")
        
        print("\n📄 First 5 rows:")
        print(self.df.head())
        
        print("\n📈 Statistical Summary for key numerical features:")
        numerical_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'parking']
        print(self.df[numerical_cols].describe())
        
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        print("No missing values!" if missing.sum() == 0 else missing[missing > 0])
        
    def preprocess_data(self):
        """Preprocess the data for linear regression"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy for preprocessing
        self.df_processed = self.df.copy()
        
        # Convert yes/no to 1/0
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea']
        
        for col in binary_cols:
            self.df_processed[col] = self.df_processed[col].map({'yes': 1, 'no': 0})
        
        # Handle furnishing status (ordinal encoding)
        furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        self.df_processed['furnishingstatus'] = self.df_processed['furnishingstatus'].map(furnishing_map)
        
        print("✅ Categorical variables converted to numerical:")
        print(f"   Binary features: {binary_cols}")
        print(f"   Furnishing status: {furnishing_map}")
        
        # Show correlations with price
        print("\n📊 Correlation with Price:")
        correlations = self.df_processed.corr()['price'].sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != 'price':
                print(f"   {feature:15}: {corr:6.3f}")
        
    def simple_linear_regression(self):
        """Implement simple linear regression (Price vs Area)"""
        print("\n" + "="*50)
        print("🔸 SIMPLE LINEAR REGRESSION: Price vs Area")
        print("="*50)
        
        # Prepare data
        X_simple = self.df_processed[['area']]
        y = self.df_processed['price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_simple, y, test_size=0.2, random_state=42
        )
        
        # Create and train the model
        self.simple_model = LinearRegression()
        self.simple_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_test = self.simple_model.predict(X_test)
        
        # Calculate metrics
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"📐 Model Equation:")
        print(f"   Price = {self.simple_model.intercept_:.0f} + {self.simple_model.coef_[0]:.2f} × Area")
        
        print(f"\n📊 Model Performance:")
        print(f"   R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
        print(f"   RMSE: ${np.sqrt(test_mse):,.0f}")
        print(f"   MAE: ${test_mae:,.0f}")
        
        # Store results for comparison
        self.simple_results = {
            'r2': test_r2,
            'rmse': np.sqrt(test_mse),
            'mae': test_mae,
            'y_test': y_test,
            'y_pred': y_pred_test
        }
        
        return y_test, y_pred_test
        
    def multiple_linear_regression(self):
        """Implement multiple linear regression"""
        print("\n" + "="*50)
        print("🔹 MULTIPLE LINEAR REGRESSION: All Features")
        print("="*50)
        
        # Prepare data
        feature_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'parking', 'prefarea', 'furnishingstatus']
        
        X_multiple = self.df_processed[feature_columns]
        y = self.df_processed['price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_multiple, y, test_size=0.2, random_state=42
        )
        
        # Create and train the model
        self.multiple_model = LinearRegression()
        self.multiple_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_test = self.multiple_model.predict(X_test)
        
        # Calculate metrics
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"📊 Model Performance:")
        print(f"   R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
        print(f"   RMSE: ${np.sqrt(test_mse):,.0f}")
        print(f"   MAE: ${test_mae:,.0f}")
        
        # Feature importance (coefficients)
        print(f"\n🏆 Most Important Features (by coefficient magnitude):")
        coefficients = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': self.multiple_model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        for i, (_, row) in enumerate(coefficients.head(6).iterrows()):
            direction = "↗️" if row['Coefficient'] > 0 else "↘️"
            print(f"   {i+1}. {row['Feature']:15}: {row['Coefficient']:8.1f} {direction}")
        
        print(f"\n📐 Model Intercept: ${self.multiple_model.intercept_:,.0f}")
        
        # Store results for comparison
        self.multiple_results = {
            'r2': test_r2,
            'rmse': np.sqrt(test_mse),
            'mae': test_mae,
            'y_test': y_test,
            'y_pred': y_pred_test
        }
        
        return y_test, y_pred_test
        
    def compare_models(self):
        """Compare simple and multiple linear regression models"""
        print("\n" + "="*50)
        print("⚖️  MODEL COMPARISON")
        print("="*50)
        
        # Create comparison
        comparison_data = {
            'Metric': ['R² Score', 'RMSE ($)', 'MAE ($)'],
            'Simple Linear': [
                f"{self.simple_results['r2']:.4f}",
                f"{self.simple_results['rmse']:,.0f}",
                f"{self.simple_results['mae']:,.0f}"
            ],
            'Multiple Linear': [
                f"{self.multiple_results['r2']:.4f}",
                f"{self.multiple_results['rmse']:,.0f}",
                f"{self.multiple_results['mae']:,.0f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Analysis
        r2_improvement = ((self.multiple_results['r2'] - self.simple_results['r2']) / self.simple_results['r2'] * 100)
        rmse_improvement = ((self.simple_results['rmse'] - self.multiple_results['rmse']) / self.simple_results['rmse'] * 100)
        
        print(f"\n📈 Performance Improvement with Multiple Features:")
        print(f"   R² improvement: {r2_improvement:+.1f}%")
        print(f"   RMSE improvement: {rmse_improvement:+.1f}%")
        
        if self.multiple_results['r2'] > self.simple_results['r2']:
            print("   🏆 Multiple Linear Regression performs better!")
        else:
            print("   ⚠️  Simple Linear Regression performs surprisingly well!")
        
    def predict_new_house(self):
        """Predict price for a sample house"""
        print("\n" + "="*50)
        print("🏠 PREDICTING SAMPLE HOUSE PRICE")
        print("="*50)
        
        # Sample house features
        sample_house = {
            'area': 7000, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
            'mainroad': 1, 'guestroom': 0, 'basement': 1, 'hotwaterheating': 0,
            'airconditioning': 1, 'parking': 2, 'prefarea': 1, 'furnishingstatus': 1
        }
        
        print("🏡 Sample House Features:")
        print(f"   Area: {sample_house['area']:,} sq ft")
        print(f"   Bedrooms: {sample_house['bedrooms']}, Bathrooms: {sample_house['bathrooms']}")
        print(f"   Stories: {sample_house['stories']}, Parking: {sample_house['parking']}")
        print(f"   Air Conditioning: {'Yes' if sample_house['airconditioning'] else 'No'}")
        print(f"   Preferred Area: {'Yes' if sample_house['prefarea'] else 'No'}")
        
        # Make predictions
        area_input = [[sample_house['area']]]
        multiple_input = [[sample_house[col] for col in ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'parking', 'prefarea', 'furnishingstatus']]]
        
        simple_pred = self.simple_model.predict(area_input)[0]
        multiple_pred = self.multiple_model.predict(multiple_input)[0]
        
        print(f"\n💰 Predicted Prices:")
        print(f"   Simple Model (area only): ${simple_pred:,.0f}")
        print(f"   Multiple Model (all features): ${multiple_pred:,.0f}")
        print(f"   Difference: ${abs(multiple_pred - simple_pred):,.0f}")
        
    def create_simple_visualization(self):
        """Create a simple visualization"""
        print("\n📊 Creating visualization...")
        
        plt.figure(figsize=(12, 4))
        
        # Simple regression plot
        plt.subplot(1, 2, 1)
        area = self.df_processed['area']
        price = self.df_processed['price']
        plt.scatter(area, price, alpha=0.5, color='lightblue', s=20)
        
        # Add regression line
        area_range = np.linspace(area.min(), area.max(), 100)
        price_pred = self.simple_model.predict(area_range.reshape(-1, 1))
        plt.plot(area_range, price_pred, color='red', linewidth=2)
        
        plt.xlabel('Area (sq ft)')
        plt.ylabel('Price ($)')
        plt.title(f'Simple Linear Regression\nR² = {self.simple_results["r2"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Actual vs Predicted for multiple regression
        plt.subplot(1, 2, 2)
        y_test = self.multiple_results['y_test']
        y_pred = self.multiple_results['y_pred']
        
        plt.scatter(y_test, y_pred, alpha=0.6, color='green', s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Multiple Linear Regression\nR² = {self.multiple_results["r2"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('linear_regression_results.png', dpi=100, bbox_inches='tight')
        print("✅ Visualization saved as 'linear_regression_results.png'")
        plt.show()
        
    def run_complete_analysis(self):
        """Run the complete linear regression analysis"""
        self.load_and_explore_data()
        self.preprocess_data()
        self.simple_linear_regression()
        self.multiple_linear_regression()
        self.compare_models()
        self.predict_new_house()
        self.create_simple_visualization()
        
        print(f"\n{'='*60}")
        print("🎉 LINEAR REGRESSION ANALYSIS COMPLETE!")
        print("="*60)
        print("\n📚 Key Concepts Demonstrated:")
        print("   ✓ Simple Linear Regression (1 feature)")
        print("   ✓ Multiple Linear Regression (12 features)")
        print("   ✓ Data preprocessing (categorical → numerical)")
        print("   ✓ Model evaluation (R², RMSE, MAE)")
        print("   ✓ Feature importance analysis")
        print("   ✓ Model comparison")
        print("   ✓ Prediction on new data")

# Run the analysis
if __name__ == "__main__":
    try:
        # Initialize the class
        housing_analysis = HousingLinearRegression('archive (1)/Housing.csv')
        
        # Run complete analysis
        housing_analysis.run_complete_analysis()
        
    except FileNotFoundError:
        print("❌ Error: Could not find the Housing.csv file.")
        print("Please make sure the file is in the 'archive (1)' folder.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
