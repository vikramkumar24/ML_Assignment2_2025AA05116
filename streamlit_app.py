"""
Streamlit App for ML Classification Model Comparison
Breast Cancer Wisconsin Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the breast cancer dataset"""
    df = pd.read_csv('breast_cancer_dataset.csv')
    # Drop any completely-empty columns and unnamed columns
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Ensure target exists (map from diagnosis if present)
    if 'target' not in df.columns and 'diagnosis' in df.columns:
        df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Keep identifier column out of features
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df


@st.cache_data
def prepare_data(df):
    """Prepare data for modeling"""
    # Ensure 'target' exists (map from 'diagnosis' if necessary)
    if 'target' not in df.columns and 'diagnosis' in df.columns:
        df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Drop any completely-empty columns and unnamed columns
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Remove identifier and non-feature columns if present
    drop_cols = [c for c in ['id', 'diagnosis'] if c in df.columns]

    # Build X and y safely
    if 'target' in df.columns:
        X = df.drop(drop_cols + ['target'], axis=1, errors='ignore')
        y = df['target']
    else:
        # Fallback: assume last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # Drop rows with missing values to avoid estimator errors
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    y = combined.iloc[:, -1]
    X = combined.drop(combined.columns[-1], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(model_name, X_train, y_train):
    """Train the selected model"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost (Gradient Boosting)': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    return model


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig


def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    return fig


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<p class="main-header">🧬 ML Classification Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("### Breast Cancer Wisconsin Dataset - Binary Classification")
    
    # Sidebar
    st.sidebar.title("📊 Model Selection")
    st.sidebar.markdown("---")
    
    model_name = st.sidebar.selectbox(
        "Choose a Machine Learning Model:",
        ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbor', 
         'Naive Bayes', 'Random Forest', 'XGBoost (Gradient Boosting)']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About Dataset:**
    - 569 samples
    - 30 features
    - Binary classification
    - Target: Malignant (0) or Benign (1)
    """)
    
    # Load and prepare data
    with st.spinner("Loading data..."):
        df = load_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Dataset overview
    st.markdown("## 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Benign Cases", sum(df['target'] == 1))
    with col4:
        st.metric("Malignant Cases", sum(df['target'] == 0))
    
    st.markdown("---")
    
    # Train model
    st.markdown(f"## 🤖 Model: {model_name}")
    
    with st.spinner(f"Training {model_name}..."):
        model = train_model(model_name, X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Display metrics
    st.markdown("### 📊 Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    with col2:
        st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
    with col3:
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    with col4:
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    with col5:
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    with col6:
        st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📉 Model Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_confusion_matrix(y_test, y_pred))
    
    with col2:
        st.pyplot(plot_roc_curve(y_test, y_pred_proba))
    
    st.markdown("---")
    
    # Comparison table
    st.markdown("## 🏆 All Models Comparison")
    
    # Load pre-computed results
    try:
        results_df = pd.read_csv('model_results.csv', index_col=0)
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Bar chart comparison
        st.markdown("### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df['Accuracy'].plot(kind='bar', color='skyblue', ax=ax)
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0.85, 1.0])
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Run ml_classification.py first to generate comparison results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Breast Cancer Wisconsin Dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
