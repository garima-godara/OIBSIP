#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Car Price Prediction Project
Author: Garima Swami
Date: 2026
Description: Machine Learning model to predict second-hand car selling prices
             based on various features like age, mileage, fuel type, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictior:
    """Class to build and train car price prediction models"""
    
    def __init__(self, csv_path):
        """
        Initialize the predictor with data
        Args:
            csv_path: Path to the car data CSV file
        """
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.label_encoders = {}
        
    def load_and_explore_data(self):
        """Load and explore the car price dataset"""
        print("\n" + "="*70)
        print("LOADING AND EXPLORING CAR PRICE DATA")
        print("="*70)
        
        self.data = pd.read_csv(self.csv_path)
        
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nFirst few records:")
        print(self.data.head(10))
        print(f"\nData Info:")
        print(self.data.info())
        print(f"\nMissing Values:")
        print(self.data.isnull().sum())
        print(f"\nStatistical Summary:")
        print(self.data.describe())
        
        print(f"\nTarget Variable Statistics (Selling Price):")
        print(f"  Mean: ₹{self.data['Selling_Price'].mean():.2f} Lakhs")
        print(f"  Median: ₹{self.data['Selling_Price'].median():.2f} Lakhs")
        print(f"  Min: ₹{self.data['Selling_Price'].min():.2f} Lakhs")
        print(f"  Max: ₹{self.data['Selling_Price'].max():.2f} Lakhs")
        print(f"  Std Dev: ₹{self.data['Selling_Price'].std():.2f} Lakhs")
        
        print(f"\nCategorical Variables Distribution:")
        print(f"\nFuel Types:")
        print(self.data['Fuel_Type'].value_counts())
        print(f"\nTransmission Types:")
        print(self.data['Transmission'].value_counts())
        print(f"\nSelling Types:")
        print(self.data['Selling_type'].value_counts())
    
    def feature_engineering(self):
        """Create new features for better prediction"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        
        # Create age feature
        self.data['Age'] = 2020 - self.data['Year']
        print("\nNew Features Created:")
        print(f"  ✓ Age (2020 - Year)")
        
        # Create depreciation ratio
        self.data['Depreciation_Ratio'] = self.data['Selling_Price'] / self.data['Present_Price']
        print(f"  ✓ Depreciation Ratio (Selling/Present Price)")
        
        # Create price per km
        self.data['Price_Per_KM'] = self.data['Selling_Price'] / (self.data['Driven_kms'] + 1)
        print(f"  ✓ Price Per KM")
        
        print(f"\nAge Statistics:")
        print(self.data['Age'].describe())
        
        print(f"\nDepreciation Ratio Statistics:")
        print(self.data['Depreciation_Ratio'].describe())
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare and preprocess data for modeling"""
        print("\n" + "="*70)
        print("DATA PREPARATION & PREPROCESSING")
        print("="*70)
        
        # Select features
        features = ['Year', 'Present_Price', 'Driven_kms', 'Owner', 'Age', 'Price_Per_KM']
        categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
        
        X = self.data[features + categorical_features].copy()
        y = self.data['Selling_Price'].copy()
        
        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            print(f"\nEncoded {col}:")
            for i, label in enumerate(le.classes_):
                print(f"  {label}: {i}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.y = y
        
        print(f"\nData Split:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        print(f"  Features: {len(features + categorical_features)}")
        print(f"\n✓ Data prepared and scaled successfully")
    
    def train_models(self):
        """Train multiple regression models"""
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        # Linear Regression
        print("\nTraining Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_model
        print("✓ Linear Regression trained")
        
        # Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("✓ Random Forest trained")
        
        # Gradient Boosting Regressor
        print("\nTraining Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
        gb_model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb_model
        print("✓ Gradient Boosting trained")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*70}")
            print(f"Model: {model_name}")
            print(f"{'─'*70}")
            
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            self.results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'predictions': y_pred_test
            }
            
            print(f"\nTraining Metrics:")
            print(f"  R² Score: {train_r2:.4f}")
            print(f"  RMSE: ₹{train_rmse:.2f} Lakhs")
            print(f"  MAE: ₹{train_mae:.2f} Lakhs")
            
            print(f"\nTest Metrics (Evaluation):")
            print(f"  R² Score: {test_r2:.4f}")
            print(f"  RMSE: ₹{test_rmse:.2f} Lakhs")
            print(f"  MAE: ₹{test_mae:.2f} Lakhs")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                print(f"\nFeature Importance:")
                feature_names = ['Year', 'Present_Price', 'Driven_kms', 'Owner', 'Age', 'Price_Per_KM', 'Fuel_Type', 'Selling_Type', 'Transmission']
                importances = model.feature_importances_
                for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {name}: {importance:.4f}")
    
    def visualize_results(self):
        """Visualize model predictions and analysis"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        sns.set_style("whitegrid")
        
        # 1. Price Distribution
        fig = plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.data['Selling_Price'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        plt.axvline(self.data['Selling_Price'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ₹{self.data['Selling_Price'].mean():.2f}L")
        plt.title('Distribution of Car Selling Prices', fontsize=12, fontweight='bold')
        plt.xlabel('Selling Price (₹ Lakhs)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Price vs Present Price
        plt.subplot(2, 3, 2)
        plt.scatter(self.data['Present_Price'], self.data['Selling_Price'], alpha=0.6, s=50)
        plt.title('Selling Price vs Present Price', fontsize=12, fontweight='bold')
        plt.xlabel('Present Price (₹ Lakhs)')
        plt.ylabel('Selling Price (₹ Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # 3. Price vs Age
        plt.subplot(2, 3, 3)
        plt.scatter(self.data['Age'], self.data['Selling_Price'], alpha=0.6, s=50, color='coral')
        plt.title('Selling Price vs Car Age', fontsize=12, fontweight='bold')
        plt.xlabel('Age (Years)')
        plt.ylabel('Selling Price (₹ Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # 4. Price vs Mileage
        plt.subplot(2, 3, 4)
        plt.scatter(self.data['Driven_kms']/1000, self.data['Selling_Price'], alpha=0.6, s=50, color='green')
        plt.title('Selling Price vs Driven KMs', fontsize=12, fontweight='bold')
        plt.xlabel('Driven KMs (thousands)')
        plt.ylabel('Selling Price (₹ Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # 5. Model Performance Comparison
        plt.subplot(2, 3, 5)
        model_names = list(self.results.keys())
        test_r2 = [self.results[m]['test_r2'] for m in model_names]
        colors = ['steelblue', 'coral', 'green']
        bars = plt.bar(model_names, test_r2, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Model Performance (R² Score)', fontsize=12, fontweight='bold')
        plt.ylabel('R² Score')
        plt.ylim([0, 1])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 6. Actual vs Predicted (Best Model)
        plt.subplot(2, 3, 6)
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])[0]
        y_pred = self.results[best_model]['predictions']
        plt.scatter(self.y_test, y_pred, alpha=0.6, s=50)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.title(f'Actual vs Predicted ({best_model})', fontsize=12, fontweight='bold')
        plt.xlabel('Actual Price (₹ Lakhs)')
        plt.ylabel('Predicted Price (₹ Lakhs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('car_price_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Main analysis saved as 'car_price_analysis.png'")
        plt.close()
        
        # Additional visualizations
        self._create_categorical_analysis()
        self._create_residual_plots()
    
    def _create_categorical_analysis(self):
        """Analyze categorical features impact on price"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Fuel Type
        fuel_price = self.data.groupby('Fuel_Type')['Selling_Price'].mean().sort_values()
        fuel_price.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_title('Average Price by Fuel Type', fontweight='bold')
        axes[0].set_xlabel('Average Price (₹ Lakhs)')
        for i, v in enumerate(fuel_price.values):
            axes[0].text(v + 0.2, i, f'₹{v:.2f}L', va='center')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Transmission
        trans_price = self.data.groupby('Transmission')['Selling_Price'].mean().sort_values()
        trans_price.plot(kind='barh', ax=axes[1], color='coral', edgecolor='black')
        axes[1].set_title('Average Price by Transmission', fontweight='bold')
        axes[1].set_xlabel('Average Price (₹ Lakhs)')
        for i, v in enumerate(trans_price.values):
            axes[1].text(v + 0.2, i, f'₹{v:.2f}L', va='center')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Seller Type
        seller_price = self.data.groupby('Selling_type')['Selling_Price'].mean().sort_values()
        seller_price.plot(kind='barh', ax=axes[2], color='green', edgecolor='black')
        axes[2].set_title('Average Price by Seller Type', fontweight='bold')
        axes[2].set_xlabel('Average Price (₹ Lakhs)')
        for i, v in enumerate(seller_price.values):
            axes[2].text(v + 0.2, i, f'₹{v:.2f}L', va='center')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Categorical analysis saved as 'categorical_analysis.png'")
        plt.close()
    
    def _create_residual_plots(self):
        """Create residual plots for error analysis"""
        best_model_name = max(self.results.items(), key=lambda x: x[1]['test_r2'])[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        y_pred = self.results[best_model_name]['predictions']
        residuals = self.y_test - y_pred
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_title(f'Residual Plot ({best_model_name})', fontweight='bold')
        axes[0].set_xlabel('Predicted Price (₹ Lakhs)')
        axes[0].set_ylabel('Residuals (₹ Lakhs)')
        axes[0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ₹{residuals.mean():.2f}L")
        axes[1].set_title('Residual Distribution', fontweight='bold')
        axes[1].set_xlabel('Residuals (₹ Lakhs)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Residual analysis saved as 'residual_analysis.png'")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*70)
        
        print("\n📊 DATASET OVERVIEW:")
        print(f"  • Total Records: {len(self.data)}")
        print(f"  • Features Used: 9")
        print(f"  • Target Variable: Selling Price")
        print(f"  • Price Range: ₹{self.data['Selling_Price'].min():.2f}L - ₹{self.data['Selling_Price'].max():.2f}L")
        
        print("\n📈 PRICE STATISTICS:")
        print(f"  • Average Price: ₹{self.data['Selling_Price'].mean():.2f} Lakhs")
        print(f"  • Median Price: ₹{self.data['Selling_Price'].median():.2f} Lakhs")
        print(f"  • Price Std Dev: ₹{self.data['Selling_Price'].std():.2f} Lakhs")
        
        print("\n📚 KEY FEATURES IMPACT:")
        print(f"  • Car Age: Inversely correlated with price")
        print(f"  • Mileage: More driven cars have lower prices")
        print(f"  • Present Price: Strongly correlated with selling price")
        print(f"  • Transmission: Automatic cars command premium")
        print(f"  • Fuel Type: Diesel cars typically priced higher")
        
        print("\n🦠 MODEL PERFORMANCE:")
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\n  Best Performing Model: {best_model[0]}")
        print(f"    ├─ Test R² Score: {best_model[1]['test_r2']:.4f}")
        print(f"    ├─ Test RMSE: ₹{best_model[1]['test_rmse']:.2f} Lakhs")
        print(f"    └─ Test MAE: ₹{best_model[1]['test_mae']:.2f} Lakhs")
        
        print("\n  All Models Performance:")
        for model_name in sorted(self.results.keys()):
            print(f"\n  {model_name}:")
            print(f"    ├─ Test R²: {self.results[model_name]['test_r2']:.4f}")
            print(f"    ├─ Test RMSE: ₹{self.results[model_name]['test_rmse']:.2f}L")
            print(f"    └─ Test MAE: ₹{self.results[model_name]['test_mae']:.2f}L")
        
        print("\n💡 KEY INSIGHTS:")
        print("  • Present price is the strongest predictor of selling price")
        print("  • Car age significantly impacts depreciation")
        print("  • Mileage is an important factor in price determination")
        print("  • Vehicle condition (Owner count) affects resale value")
        print("  • Transmission type impacts price premium/discount")
        
        print("\n📁 GENERATED FILES:")
        print("  • car_price_analysis.png - Main price analysis visualizations")
        print("  • categorical_analysis.png - Impact of categorical features")
        print("  • residual_analysis.png - Model error analysis")
        
        print("\n" + "="*70)
        print("Car Price Prediction Model Complete!")
        print("="*70 + "\n")

def main():
    """Main execution function"""
    print("\n\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "CAR PRICE PREDICTION - MACHINE LEARNING" + " "*14 + "#")
    print("#" + " "*12 + "Second-hand Car Resale Price Prediction" + " "*16 + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    # Initialize predictor
    predictor = CarPricePredictior('car_data.csv')
    
    # Execute pipeline
    predictor.load_and_explore_data()
    predictor.feature_engineering()
    predictor.prepare_data()
    predictor.train_models()
    predictor.evaluate_models()
    predictor.visualize_results()
    predictor.generate_report()

if __name__ == "__main__":
    main()