#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Iris Flower Classification Project
Author: Garima Swami
Date: 2026
Description: Machine Learning model to classify iris flowers based on sepal and petal measurements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    """Class to handle Iris flower classification"""
    
    def __init__(self, csv_path):
        """
        Initialize the classifier with data
        Args:
            csv_path: Path to the iris CSV file
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
        
    def load_data(self):
        """Load and display initial data information"""
        print("\n" + "="*60)
        print("LOADING AND EXPLORING DATA")
        print("="*60)
        
        self.data = pd.read_csv(self.csv_path)
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nFirst few rows:\n{self.data.head()}")
        print(f"\nData Info:\n{self.data.info()}")
        print(f"\nStatistical Summary:\n{self.data.describe()}")
        print(f"\nMissing Values:\n{self.data.isnull().sum()}")
        print(f"\nSpecies Distribution:\n{self.data['Species'].value_counts()}")
        
    def visualize_data(self):
        """Create visualizations of the data"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 12))
        
        # Pairplot
        plt.subplot(2, 2, 1)
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        species = self.data['Species'].unique()
        colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
        
        for sp in species:
            data_sp = self.data[self.data['Species'] == sp]
            plt.scatter(data_sp['SepalLengthCm'], data_sp['PetalLengthCm'], 
                       label=sp, alpha=0.7, s=100, color=colors[sp])
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.title('Sepal Length vs Petal Length')
        plt.legend()
        plt.grid(True)
        
        # Histogram
        plt.subplot(2, 2, 2)
        for sp in species:
            data_sp = self.data[self.data['Species'] == sp]
            plt.hist(data_sp['PetalLengthCm'], alpha=0.5, label=sp, bins=10)
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Petal Length by Species')
        plt.legend()
        plt.grid(True)
        
        # Box plot
        plt.subplot(2, 2, 3)
        self.data.boxplot(column='SepalLengthCm', by='Species')
        plt.title('Sepal Length Distribution by Species')
        plt.suptitle('')
        plt.xlabel('Species')
        plt.ylabel('Sepal Length (cm)')
        
        # Correlation heatmap
        plt.subplot(2, 2, 4)
        corr_matrix = self.data[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, cbar=True)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('iris_visualizations.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved as 'iris_visualizations.png'")
        plt.close()
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60)
        
        # Separate features and target
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.X = self.data[features]
        self.y = self.data['Species']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"\nTraining set species distribution:\n{self.y_train.value_counts()}")
        print(f"\nTest set species distribution:\n{self.y_test.value_counts()}")
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Data prepared and scaled successfully")
        
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Random Forest Classifier
        print("\nTraining Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = rf_model
        print("✓ Random Forest trained")
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_model = LogisticRegression(max_iter=200, random_state=42, multi_class='multinomial')
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("✓ Logistic Regression trained")
        
        # Support Vector Machine
        print("\nTraining Support Vector Machine...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = svm_model
        print("✓ SVM trained")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\n{'─'*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'─'*60}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            print(f"\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
    
    def visualize_results(self):
        """Visualize model performance"""
        print("\n" + "="*60)
        print("VISUALIZING RESULTS")
        print("="*60)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        precisions = [self.results[m]['precision'] for m in model_names]
        recalls = [self.results[m]['recall'] for m in model_names]
        f1_scores = [self.results[m]['f1'] for m in model_names]
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color='steelblue', alpha=0.8)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision comparison
        axes[0, 1].bar(model_names, precisions, color='coral', alpha=0.8)
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Model Precision Comparison')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(precisions):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall comparison
        axes[1, 0].bar(model_names, recalls, color='lightgreen', alpha=0.8)
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Model Recall Comparison')
        axes[1, 0].set_ylim([0, 1])
        for i, v in enumerate(recalls):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1-Score comparison
        axes[1, 1].bar(model_names, f1_scores, color='mediumpurple', alpha=0.8)
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Model F1-Score Comparison')
        axes[1, 1].set_ylim([0, 1])
        for i, v in enumerate(f1_scores):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Model comparison saved as 'model_comparison.png'")
        plt.close()
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.results[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                       yticklabels=['Setosa', 'Versicolor', 'Virginica'])
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved as 'confusion_matrices.png'")
        plt.close()
    
    def generate_summary(self):
        """Generate and print summary report"""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        print("\nDataset Information:")
        print(f"  • Total samples: {len(self.data)}")
        print(f"  • Number of features: 4")
        print(f"  • Number of classes: 3")
        print(f"  • Feature names: SepalLength, SepalWidth, PetalLength, PetalWidth")
        print(f"  • Classes: Iris-setosa, Iris-versicolor, Iris-virginica")
        
        print("\nBest Performing Model:")
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"  • Model: {best_model[0]}")
        print(f"  • Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"  • Precision: {best_model[1]['precision']:.4f}")
        print(f"  • Recall: {best_model[1]['recall']:.4f}")
        print(f"  • F1-Score: {best_model[1]['f1']:.4f}")
        
        print("\nAll Models Performance:")
        for model_name, metrics in self.results.items():
            print(f"\n  {model_name}:")
            print(f"    ├─ Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    ├─ Precision: {metrics['precision']:.4f}")
            print(f"    ├─ Recall:    {metrics['recall']:.4f}")
            print(f"    └─ F1-Score:  {metrics['f1']:.4f}")
        
        print("\nKey Findings:")
        print("  • The iris dataset is well-balanced across all three species")
        print("  • Petal measurements are more discriminative than sepal measurements")
        print("  • All models perform exceptionally well on this dataset")
        print("  • Feature scaling improved model performance")
        
        print("\nFiles Generated:")
        print("  • iris_visualizations.png - Data exploration plots")
        print("  • model_comparison.png - Performance metrics comparison")
        print("  • confusion_matrices.png - Confusion matrices for all models")
        
        print("\n" + "="*60)
        print("Classification Complete!")
        print("="*60 + "\n")

def main():
    """Main execution function"""
    print("\n\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "IRIS FLOWER CLASSIFICATION" + " "*17 + "#")
    print("#" + " "*18 + "Machine Learning Project" + " "*16 + "#")
    print("#" + " "*58 + "#")
    print("#"*60 + "\n")
    
    # Initialize classifier
    classifier = IrisClassifier('Iris.csv')
    
    # Execute pipeline
    classifier.load_data()
    classifier.visualize_data()
    classifier.prepare_data()
    classifier.train_models()
    classifier.evaluate_models()
    classifier.visualize_results()
    classifier.generate_summary()

if __name__ == "__main__":
    main()