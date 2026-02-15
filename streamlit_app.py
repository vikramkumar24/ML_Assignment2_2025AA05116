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
    try:
        df = pd.read_csv('breast_cancer_dataset.csv')
        return df
    except FileNotFoundError:
        return None


def validate_dataset(df):
    """Validate uploaded dataset"""
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
    
    return errors


@st.cache_data
def prepare_data(df):
    """Prepare data for modeling"""
    X = df.drop('target', axis=1)
    y = df['target']
    
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
    """Calculate all evaluation metrics for binary or multiclass"""
    n_classes = len(np.unique(y_true))
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    
    # Handle binary vs multiclass
    if n_classes == 2:
        # Binary classification - use probabilities for positive class
        if len(y_pred_proba.shape) > 1:
            y_pred_proba_binary = y_pred_proba[:, 1]
        else:
            y_pred_proba_binary = y_pred_proba
            
        metrics.update({
            'AUC Score': roc_auc_score(y_true, y_pred_proba_binary),
            'Precision': precision_score(y_true, y_pred, average='binary'),
            'Recall': recall_score(y_true, y_pred, average='binary'),
            'F1 Score': f1_score(y_true, y_pred, average='binary')
        })
    else:
        # Multiclass classification - use weighted average
        try:
            metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba, 
                                                  multi_class='ovr', average='weighted')
        except:
            metrics['AUC Score'] = 0.0  # If AUC calculation fails
            
        metrics.update({
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        })
    
    return metrics



def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(np.unique(y_true))
    
    # Create dynamic labels
    class_labels = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig


def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve - binary only"""
    n_classes = len(np.unique(y_true))
    
    if n_classes > 2:
        # For multiclass, show a message instead
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.text(0.5, 0.5, 'ROC Curve\nNot available for\nmulticlass classification', 
                ha='center', va='center', fontsize=16)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Binary Only)')
        return fig
    
    # Binary classification
    if len(y_pred_proba.shape) > 1:
        y_pred_proba_binary = y_pred_proba[:, 1]
    else:
        y_pred_proba_binary = y_pred_proba
        
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba_binary)
    auc_score = roc_auc_score(y_true, y_pred_proba_binary)
    
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
    st.markdown("### Machine Learning Classification - Binary/Multiclass")
    
    # Sidebar - Dataset Source
    st.sidebar.title("📁 Dataset Source")
    st.sidebar.markdown("---")
    
    dataset_source = st.sidebar.radio(
        "Choose dataset:",
        ["Use Default Dataset", "Upload CSV File"]
    )
    
    uploaded_file = None
    df = None
    dataset_name = "Breast Cancer Wisconsin"
    
    if dataset_source == "Upload CSV File":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📋 Dataset Requirements:**")
        st.sidebar.info("""
        - Must have a 'target' column
        - Minimum 12 features
        - Minimum 500 instances
        - All features must be numeric
        - Target can be binary or multiclass
        """)
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file with features and a 'target' column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name.replace('.csv', '')
                
                # Validate dataset
                errors = validate_dataset(df)
                
                if errors:
                    st.sidebar.error("**Dataset Validation Failed:**")
                    for error in errors:
                        st.sidebar.error(error)
                    st.error("Please upload a valid dataset that meets the requirements.")
                    st.stop()
                else:
                    st.sidebar.success("✅ Dataset validated successfully!")
                    
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                st.error("Failed to load the uploaded file. Please check the format.")
                st.stop()
        else:
            st.info("👆 Please upload a CSV file from the sidebar to get started.")
            st.markdown("---")
            st.markdown("### 📋 Sample Dataset Format")
            st.markdown("""
            Your CSV file should look like this:
            
            | feature_1 | feature_2 | feature_3 | ... | feature_n | target |
            |-----------|-----------|-----------|-----|-----------|--------|
            | 0.5       | 1.2       | 3.4       | ... | 2.1       | 0      |
            | 0.8       | 1.5       | 2.9       | ... | 3.2       | 1      |
            
            **Requirements:**
            - ✅ At least 12 numeric features
            - ✅ At least 500 rows
            - ✅ A 'target' column with class labels (0, 1, 2, etc.)
            - ✅ No missing values or handle them before upload
            """)
            st.stop()
    else:
        # Load default dataset
        with st.spinner("Loading default dataset..."):
            df = load_data()
            if df is None:
                st.error("Default dataset not found. Please upload your own CSV file.")
                st.stop()
    
    # Sidebar - Model Selection
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Model Selection")
    st.sidebar.markdown("---")
    
    model_name = st.sidebar.selectbox(
        "Choose a Machine Learning Model:",
        ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbor', 
         'Naive Bayes', 'Random Forest', 'XGBoost (Gradient Boosting)']
    )
    
    st.sidebar.markdown("---")
    
    # Dataset info in sidebar
    feature_count = len([col for col in df.columns if col != 'target'])
    target_classes = df['target'].nunique()
    class_type = "Binary" if target_classes == 2 else f"Multiclass ({target_classes} classes)"
    
    st.sidebar.info(f"""
    **Dataset: {dataset_name}**
    - {len(df)} samples
    - {feature_count} features
    - {class_type} classification
    """)
    
    # Load and prepare data
    with st.spinner("Preparing data..."):
        X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Dataset overview
    st.markdown("## 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    target_counts = df['target'].value_counts().sort_index()
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric(f"Class 0", int(target_counts.iloc[0]))
    with col4:
        if len(target_counts) > 1:
            st.metric(f"Class 1", int(target_counts.iloc[1]))
        else:
            st.metric("Classes", len(target_counts))
    
    # Show dataset preview
    with st.expander("👁️ View Dataset Preview"):
        st.dataframe(df.head(10))
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show class distribution
        st.markdown("**Class Distribution:**")
        class_dist = df['target'].value_counts().sort_index()
        for cls, count in class_dist.items():
            percentage = (count / len(df)) * 100
            st.write(f"- Class {cls}: {count} samples ({percentage:.1f}%)")
    
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
    
    # Add sample dataset download
    st.markdown("---")
    st.markdown("## 📥 Need Sample Data?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Download Sample CSV Template"):
            # Create a sample CSV
            sample_data = {
                'feature_1': np.random.rand(100),
                'feature_2': np.random.rand(100),
                'feature_3': np.random.rand(100),
                'feature_4': np.random.rand(100),
                'feature_5': np.random.rand(100),
                'feature_6': np.random.rand(100),
                'feature_7': np.random.rand(100),
                'feature_8': np.random.rand(100),
                'feature_9': np.random.rand(100),
                'feature_10': np.random.rand(100),
                'feature_11': np.random.rand(100),
                'feature_12': np.random.rand(100),
                'target': np.random.randint(0, 2, 100)
            }
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Sample Template",
                data=csv,
                file_name="sample_classification_data.csv",
                mime="text/csv"
            )
    
    with col2:
        st.info("Use this template to understand the required CSV format. Modify it with your own data!")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Dataset: {dataset_name}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
