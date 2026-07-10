# Task 1: Iris Flower Classification

## 📋 Project Overview

This project implements a **Machine Learning classification system** to predict iris flower species based on sepal and petal measurements. The iris dataset is a classic benchmark dataset in machine learning, containing measurements of iris flowers from three different species.

### 🎯 Objective
Build and train machine learning models that can accurately classify iris flowers into three species:
- **Iris Setosa**
- **Iris Versicolor** 
- **Iris Virginica**

---

## 📊 Dataset Information

### Dataset Details
- **Total Samples:** 150 iris flowers
- **Features:** 4 numerical measurements
- **Classes:** 3 iris species
- **Samples per Class:** 50 (balanced dataset)

### Features
1. **SepalLengthCm** - Length of the sepal in centimeters
2. **SepalWidthCm** - Width of the sepal in centimeters
3. **PetalLengthCm** - Length of the petal in centimeters
4. **PetalWidthCm** - Width of the petal in centimeters

### Target Variable
- **Species** - One of three iris species (Setosa, Versicolor, Virginica)

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

---

## 📁 Project Structure

```
Task 1 - Garima Swami - Iris Flower Classification/
├── Iris.csv                          # Dataset file
├── iris_classification.py            # Main Python script
├── README.md                         # Project documentation
├── iris_visualizations.png          # Generated data exploration plots
├── model_comparison.png             # Generated performance comparison
└── confusion_matrices.png           # Generated confusion matrices
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
python iris_classification.py
```

### Expected Output
The script will:
1. Load and explore the iris dataset
2. Generate visualizations of the data
3. Prepare and scale the data
4. Train three different classification models
5. Evaluate each model with detailed metrics
6. Generate performance comparison visualizations
7. Display a comprehensive summary report

---

## 🤖 Models Implemented

### 1. Random Forest Classifier
- **Type:** Ensemble Learning
- **Parameters:** 100 estimators
- **Advantages:** Handles non-linear relationships, feature importance analysis

### 2. Logistic Regression
- **Type:** Linear Classification
- **Parameters:** Multi-class classification with multinomial approach
- **Advantages:** Simple, interpretable, fast training

### 3. Support Vector Machine (SVM)
- **Type:** Kernel-based Classification
- **Parameters:** RBF kernel, probability calibration
- **Advantages:** Excellent for multi-class problems, handles non-linearity

---

## 📈 Model Evaluation Metrics

Each model is evaluated using:

1. **Accuracy** - Overall correctness of predictions
2. **Precision** - Accuracy of positive predictions
3. **Recall** - Coverage of actual positives
4. **F1-Score** - Harmonic mean of precision and recall
5. **Confusion Matrix** - Detailed prediction breakdown
6. **Classification Report** - Per-class metrics

---

## 🔍 Key Findings

### Data Characteristics
- ✅ **Balanced Dataset** - Each species has exactly 50 samples
- ✅ **No Missing Values** - Complete dataset with no data quality issues
- ✅ **Well-Separated Classes** - Species are clearly distinguishable

### Feature Importance
- **PetalLength** and **PetalWidth** are highly discriminative
- **SepalLength** and **SepalWidth** provide additional classification power
- Strong correlation between petal measurements

### Model Performance
- All three models achieve **>95% accuracy**
- Random Forest and SVM show superior performance
- Logistic Regression provides fast, interpretable results
- Excellent generalization on test set

---

## 📊 Generated Visualizations

### 1. iris_visualizations.png
Includes:
- Scatter plot: Sepal Length vs Petal Length
- Histogram: Petal Length distribution by species
- Box plot: Sepal Length distribution
- Correlation heatmap: Feature relationships

### 2. model_comparison.png
Shows:
- Accuracy comparison across models
- Precision scores for all models
- Recall metrics comparison
- F1-Score comparison

### 3. confusion_matrices.png
Displays:
- Confusion matrix for Random Forest
- Confusion matrix for Logistic Regression
- Confusion matrix for SVM

---

## 💡 Learning Outcomes

This project demonstrates:

1. **Data Loading & Exploration**
   - Loading CSV files with pandas
   - Exploratory data analysis
   - Statistical summaries

2. **Data Preprocessing**
   - Train-test splitting with stratification
   - Feature scaling and standardization
   - Handling imbalanced data

3. **Model Building**
   - Implementing multiple classification algorithms
   - Hyperparameter tuning
   - Model comparison

4. **Evaluation & Validation**
   - Computing evaluation metrics
   - Analyzing confusion matrices
   - Interpreting classification reports

5. **Visualization**
   - Creating informative plots
   - Comparing model performance visually
   - Presenting results professionally

---

## 🔧 Customization

### Modify Train-Test Split
Change the `test_size` parameter in `prepare_data()` method:
```python
classifier.prepare_data(test_size=0.3)  # 30% test, 70% train
```

### Adjust Model Parameters
Edit the model initialization sections:
```python
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
```

### Add New Models
Extend the `train_models()` method:
```python
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
gb_model.fit(self.X_train_scaled, self.y_train)
self.models['Gradient Boosting'] = gb_model
```

---

## 📝 Notes

- The dataset is well-suited for learning because it's clean and balanced
- Feature scaling is important for distance-based models (KNN, SVM)
- Cross-validation can be implemented for more robust evaluation
- The small dataset size makes it ideal for educational purposes

---

## 📚 References

- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) - Wikipedia
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Basics](https://www.coursera.org/learn/machine-learning)

---

## 👨‍💻 Author

**Garima Swami**  
*Data Science Internship Project*  
*Date: 2026*

---

## ✨ Conclusion

This project successfully demonstrates the complete machine learning pipeline from data loading to model evaluation. The iris flower classification problem serves as an excellent introduction to supervised learning and classification tasks. The high accuracy achieved by multiple models confirms the dataset's suitability for this task.

**Happy Learning!** 🎓
