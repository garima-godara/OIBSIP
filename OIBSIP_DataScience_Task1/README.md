# Task 1: Iris Flower Classification

## 📋 Project Overview

This project implements a **Machine Learning classification system** to predict iris flower species based on sepal and petal measurements. The iris dataset is a classic benchmark dataset in machine learning, containing measurements of iris flowers from three different species.

### 🎯 Objective
Build and train machine learning models that can accurately classify iris flowers into three species:
- **Iris Setosa**
- **Iris Versicolor** - **Iris Virginica**

### 📝 Steps Performed
1. **Data Loading & Exploration:** Loaded the dataset and checked for missing values or imbalances.
2. **Data Visualization:** Created pairplots, histograms, and heatmaps to understand feature distributions and correlations.
3. **Data Preprocessing:** Split the data into training and testing sets (80/20) and applied feature scaling.
4. **Model Training:** Trained three different algorithms (Random Forest, Logistic Regression, and Support Vector Machine).
5. **Model Evaluation:** Evaluated the models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

### 🛠️ Tools Used
- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

### 💡 Outcome
Successfully built and evaluated multiple classification models. All three models (Random Forest, Logistic Regression, SVM) achieved **>95% accuracy** on the test dataset, demonstrating that petal and sepal measurements are highly effective features for distinguishing between the three Iris species.

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

## 📁 Project Structure
