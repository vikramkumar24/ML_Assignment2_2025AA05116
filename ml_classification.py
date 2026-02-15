"""
Machine Learning Classification Assignment
Breast Cancer Wisconsin Dataset - Binary Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare the breast cancer dataset"""
    # Load dataset
    df = pd.read_csv('breast_cancer_dataset.csv')
    # Drop any completely-empty columns (e.g. trailing comma created unnamed col)
    df = df.dropna(axis=1, how='all')
    # Drop columns with names like 'Unnamed' that sometimes appear
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Map diagnosis to binary target if needed and drop unused columns
    if 'target' not in df.columns and 'diagnosis' in df.columns:
        df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Remove identifier and non-feature columns
    drop_cols = [c for c in ['id', 'diagnosis'] if c in df.columns]
    X = df.drop(drop_cols + ['target'], axis=1)
    y = df['target']

    # Combine and drop rows with missing values then split back
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    y = combined['target']
    X = combined.drop('target', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all required evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1': f1_score(y_true, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def train_and_evaluate_models():
    """Train all 6 models and evaluate them"""
    
    # Load data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Store results
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[model_name] = metrics
        
        # Print results
        print(f"{model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print()
    
    return results


def create_results_table(results):
    """Create a formatted results table"""
    df_results = pd.DataFrame(results).T
    df_results = df_results.round(4)
    return df_results


if __name__ == "__main__":
    print("="*70)
    print("ML Classification Assignment - Breast Cancer Dataset")
    print("="*70)
    print()
    
    # Train and evaluate all models
    results = train_and_evaluate_models()
    
    # Create results table
    results_df = create_results_table(results)
    
    print("="*70)
    print("FINAL RESULTS TABLE")
    print("="*70)
    print(results_df)
    print()
    
    # Save results
    results_df.to_csv('model_results.csv')
    print("Results saved to 'model_results.csv'")
