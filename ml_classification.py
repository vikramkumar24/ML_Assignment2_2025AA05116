"""
Machine Learning Classification Assignment
Supports both default dataset and custom CSV uploads
"""

import pandas as pd
import numpy as np
import sys
import os
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


def validate_dataset(df):
    """Validate the dataset meets requirements"""
    errors = []
    
    # Check if dataset has target column
    if 'target' not in df.columns:
        errors.append("❌ Dataset must have a 'target' column")
    
    # Check minimum features (excluding target)
    feature_cols = [col for col in df.columns if col != 'target']
    if len(feature_cols) < 12:
        errors.append(f"❌ Dataset must have at least 12 features (found {len(feature_cols)})")
    
    # Check minimum instances
    if len(df) < 500:
        errors.append(f"❌ Dataset must have at least 500 instances (found {len(df)})")
    
    # Check if target is binary or multiclass
    unique_targets = df['target'].nunique()
    if unique_targets < 2:
        errors.append("❌ Target must have at least 2 classes")
    
    # Check for non-numeric columns (except target)
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        errors.append(f"❌ All features must be numeric. Non-numeric columns: {', '.join(non_numeric)}")
    
    # Check for missing values
    if df.isnull().any().any():
        errors.append("❌ Dataset contains missing values. Please handle them before training.")
    
    return errors


def load_and_prepare_data(csv_file='breast_cancer_dataset.csv'):
    """Load and prepare the dataset from CSV file"""
    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file '{csv_file}' not found!")
    
    # Load dataset
    print(f"Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Validate dataset
    print("Validating dataset...")
    errors = validate_dataset(df)
    
    if errors:
        print("\n❌ Dataset validation failed:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    
    print("✅ Dataset validation passed!")
    
    # Display dataset info
    feature_cols = [col for col in df.columns if col != 'target']
    n_classes = df['target'].nunique()
    class_type = "Binary" if n_classes == 2 else f"Multiclass ({n_classes} classes)"
    
    print(f"\nDataset Info:")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Classification Type: {class_type}")
    print(f"  - Class Distribution:")
    for cls, count in df['target'].value_counts().sort_index().items():
        percentage = (count / len(df)) * 100
        print(f"    Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all required evaluation metrics for binary or multiclass"""
    n_classes = len(np.unique(y_true))
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Handle binary vs multiclass
    if n_classes == 2:
        # Binary classification
        if len(y_pred_proba.shape) > 1:
            y_pred_proba_binary = y_pred_proba[:, 1]
        else:
            y_pred_proba_binary = y_pred_proba
            
        metrics.update({
            'AUC': roc_auc_score(y_true, y_pred_proba_binary),
            'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        })
    else:
        # Multiclass classification - use weighted average
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                           multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = 0.0  # If AUC calculation fails
            
        metrics.update({
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        })
    
    return metrics


def train_and_evaluate_models(csv_file='breast_cancer_dataset.csv'):
    """Train all 6 models and evaluate them"""
    
    # Load data
    print("="*70)
    X_train, X_test, y_train, y_test, df = load_and_prepare_data(csv_file)
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}\n")
    
    n_classes = len(np.unique(y_train))
    
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
        y_pred_proba = model.predict_proba(X_test)
        
        # For binary classification, extract positive class probabilities
        if n_classes == 2 and len(y_pred_proba.shape) > 1:
            y_pred_proba_for_metrics = y_pred_proba[:, 1]
        else:
            y_pred_proba_for_metrics = y_pred_proba
        
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
    print("ML Classification Assignment")
    print("="*70)
    print()
    
    # Check if CSV file is provided as command-line argument
    csv_file = 'breast_cancer_dataset.csv'  # Default
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Using custom dataset: {csv_file}\n")
    else:
        print("Using default dataset: breast_cancer_dataset.csv")
        print("Tip: You can provide a custom CSV file: python ml_classification.py your_data.csv\n")
    
    # Train and evaluate all models
    try:
        results = train_and_evaluate_models(csv_file)
        
        # Create results table
        results_df = create_results_table(results)
        
        print("="*70)
        print("FINAL RESULTS TABLE")
        print("="*70)
        print(results_df)
        print()
        
        # Save results
        output_file = csv_file.replace('.csv', '_results.csv')
        results_df.to_csv(output_file)
        print(f"Results saved to '{output_file}'")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure your CSV file exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)

