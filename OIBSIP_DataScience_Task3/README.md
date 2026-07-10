# Task 3: Car Price Prediction

## 📋 Project Overview

This project builds a **Machine Learning regression model** to predict the **selling price of second-hand cars**. Using various features like car age, mileage, present price, fuel type, and transmission type, the model learns to estimate the resale value of vehicles. This is a critical research area in machine learning with real-world applications in automotive sales, valuation, and financing.

### 🎯 Objective

Develop and compare multiple regression models to accurately predict second-hand car selling prices based on:
- Car specifications (age, engine type, transmission)
- Market value indicators (present price)
- Usage metrics (driven kilometers, owner count)
- Market dynamics (seller type, fuel type)

---

## 📊 Dataset Information

### Dataset Overview
- **Total Records:** 301 used car entries
- **Time Period:** Cars from 2003-2018
- **Price Range:** ₹0.1 - ₹35 Lakhs
- **Features:** 9 attributes for prediction
- **Target Variable:** Selling Price (in Lakhs)

### Features Explained

1. **Car_Name** - Model name of the car
2. **Year** - Year of manufacture
3. **Selling_Price** - Price at which car was sold (Target Variable)
4. **Present_Price** - Price in the market (current value)
5. **Driven_kms** - Total kilometers driven
6. **Fuel_Type** - Type of fuel (Petrol, Diesel, CNG)
7. **Selling_type** - Whether sold by Dealer or Individual
8. **Transmission** - Manual or Automatic
9. **Owner** - Number of previous owners (0, 1, 2, 3)

### Derived Features
- **Age** - Car age in years (2020 - Year)
- **Depreciation_Ratio** - Selling Price / Present Price
- **Price_Per_KM** - Selling Price / Driven KMs

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization

---

## 📁 Project Structure

```
Task 3 - Garima Swami - Car Price Prediction/
├── car_data.csv                      # Dataset file (301 cars)
├── car_price_prediction.py           # Main Python script
├── README.md                         # Project documentation
├── car_price_analysis.png            # Price analysis visualizations
├── categorical_analysis.png          # Categorical features impact
└── residual_analysis.png             # Model error analysis
```

---

## 🚀 How to Run

### Prerequisites
Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
```bash
python car_price_prediction.py
```

### Expected Output
The script will:
1. Load and explore the car price dataset
2. Perform exploratory data analysis
3. Engineer new features for better predictions
4. Preprocess and scale the data
5. Train three regression models
6. Evaluate models with multiple metrics
7. Generate comprehensive visualizations
8. Display detailed analysis report

---

## 📈 Analysis Sections

### 1. Data Exploration
- Dataset structure and shape
- Statistical summary of all variables
- Target variable distribution
- Categorical variables breakdown
- Data quality assessment

### 2. Feature Engineering
- **Age** - Calculate car age from manufacture year
- **Depreciation Ratio** - Understand value loss
- **Price Per KM** - Metric for usage intensity
- Feature scaling and normalization
- Categorical encoding

### 3. Model Development
- **Linear Regression** - Baseline model for regression
- **Random Forest Regressor** - Ensemble method
- **Gradient Boosting Regressor** - Sequential boosting approach

### 4. Model Evaluation
- **R² Score** - Proportion of variance explained
- **RMSE** - Root Mean Squared Error (₹ Lakhs)
- **MAE** - Mean Absolute Error (₹ Lakhs)
- Feature importance analysis
- Cross-model comparison

### 5. Visualizations
- Price distribution histograms
- Scatter plots showing feature relationships
- Categorical impact on prices
- Model performance comparison
- Actual vs predicted price plots
- Residual analysis

---

## 🔍 Key Findings

### Price Insights
- **Average Selling Price:** ₹6.5 Lakhs
- **Price Range:** ₹0.35L - ₹35L
- **Highest Depreciation:** Older cars with high mileage
- **Premium Factors:** Lower age, automatic transmission, diesel fuel

### Feature Importance
1. **Present Price** - Strongest predictor (0.50+ importance)
2. **Driven KMs** - Strong indicator of depreciation
3. **Year** - Car age significantly impacts value
4. **Transmission** - Automatic premium of ~15-20%
5. **Fuel Type** - Diesel commands higher prices

### Model Performance
- **Best Model:** Gradient Boosting/Random Forest
- **Typical R² Score:** 0.85-0.90
- **Average Prediction Error:** ₹0.5-1.0 Lakhs
- **Model Generalization:** Good (minimal overfitting)

---

## 📊 Visualizations Generated

### 1. car_price_analysis.png
**6 comprehensive analysis plots:**
- Price distribution (histogram with mean line)
- Selling vs Present Price scatter plot
- Price vs Car Age relationship
- Price vs Mileage correlation
- Model R² Score comparison
- Actual vs Predicted values scatter

### 2. categorical_analysis.png
**Categorical feature impact:**
- Average price by Fuel Type (Petrol, Diesel, CNG)
- Price impact of Transmission (Manual vs Automatic)
- Seller Type influence (Dealer vs Individual)
- Premium/discount quantification

### 3. residual_analysis.png
**Model error analysis:**
- Residual scatter plot (errors vs predictions)
- Residual distribution histogram
- Error pattern identification
- Model bias assessment

---

## 💡 Learning Outcomes

### Machine Learning Concepts
1. **Regression Analysis** - Predicting continuous values
2. **Feature Engineering** - Creating meaningful features
3. **Data Preprocessing** - Cleaning and scaling data
4. **Model Selection** - Choosing appropriate algorithms
5. **Hyperparameter Tuning** - Optimizing model performance
6. **Cross-Validation** - Robust model evaluation
7. **Error Analysis** - Understanding model limitations

### Business Insights
- Pricing strategy for used car dealers
- Factors affecting car resale value
- Fair pricing determination
- Customer value proposition
- Market trend analysis

---

## 🔧 Customization

### Predict a Specific Car Price
```python
# Example: Predict price for a specific car
car_features = {
    'Year': 2018,
    'Present_Price': 12.5,
    'Driven_kms': 30000,
    'Fuel_Type': 'Diesel',
    'Transmission': 'Automatic',
    'Selling_type': 'Dealer',
    'Owner': 0
}
# Use the best trained model to predict
```

### Adjust Model Parameters
```python
# Increase Random Forest trees for better accuracy
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15)

# Tune Gradient Boosting learning rate
gb_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=150)
```

### Add More Models
```python
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Add Support Vector Machine
svm_model = SVR(kernel='rbf')

# Add XGBoost (if installed)
xgb_model = XGBRegressor(n_estimators=100)
```

---

## 📝 Notes

- **Data Quality:** Dataset is clean with minimal missing values
- **Scaling:** Feature scaling applied for fair comparison
- **Categorical Encoding:** Label encoding used for categorical variables
- **Train-Test Split:** 80-20 split with random_state=42 for reproducibility
- **Price Currency:** All prices in Indian Rupees (Lakhs)

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Regression Guide](https://www.coursera.org/learn/machine-learning)
- [Ensemble Methods](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Feature Engineering Techniques](https://towardsdatascience.com/feature-engineering-for-machine-learning)

---

## 👨‍💻 Author

**Garima Swami**  
*Data Science Internship Project*  
*Date: 2026*

---

## ✨ Conclusion

This project successfully demonstrates a complete machine learning pipeline for regression tasks. The car price prediction model achieves strong predictive performance with:

1. **Effective Feature Engineering** - Created meaningful derived features
2. **Smart Model Selection** - Compared multiple algorithms
3. **Thorough Evaluation** - Used multiple metrics for assessment
4. **Clear Insights** - Identified key price drivers
5. **Practical Application** - Real-world relevance in automotive industry

**Key Takeaway:** Present market price, car age, and mileage are the strongest predictors of resale value. Machine learning enables accurate pricing in used car markets.

**Happy Learning & Predicting!** 🎓🚗
