# Machine Learning Classification Assignment
## Breast Cancer Wisconsin Dataset

---

## Problem Statement

The objective of this assignment is to develop and evaluate multiple machine learning classification models to predict whether a breast tumor is **malignant** (cancerous) or **benign** (non-cancerous) based on various cellular characteristics extracted from digitized images of fine needle aspirate (FNA) of breast masses.

This is a **binary classification problem** where:
- **Class 0**: Malignant (cancerous)
- **Class 1**: Benign (non-cancerous)

Early and accurate detection of breast cancer is crucial for effective treatment and improved patient outcomes. This project aims to compare the performance of six different machine learning algorithms to identify the most effective model for this medical diagnosis task.

---

## Dataset Description

**Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset

**Source**: UCI Machine Learning Repository / Scikit-learn library

**Dataset Characteristics**:
- **Number of Instances**: 569
- **Number of Features**: 30 (all numeric, real-valued)
- **Target Variable**: Binary (Malignant: 0, Benign: 1)
- **Class Distribution**:
  - Benign (1): 357 samples (62.7%)
  - Malignant (0): 212 samples (37.3%)

**Features**:
The dataset contains 30 features computed from digitized images of breast mass cell nuclei. These features describe characteristics of the cell nuclei present in the image. The features include:

1. **radius** - mean of distances from center to points on the perimeter
2. **texture** - standard deviation of gray-scale values
3. **perimeter** - perimeter of the nucleus
4. **area** - area of the nucleus
5. **smoothness** - local variation in radius lengths
6. **compactness** - perimeter² / area - 1.0
7. **concavity** - severity of concave portions of the contour
8. **concave points** - number of concave portions of the contour
9. **symmetry** - symmetry of the nucleus
10. **fractal dimension** - "coastline approximation" - 1

For each of these 10 characteristics, three values are computed:
- Mean
- Standard Error
- "Worst" (mean of the three largest values)

This results in 30 features in total.

**Data Split**:
- Training Set: 80% (455 samples)
- Test Set: 20% (114 samples)
- Stratified split to maintain class distribution

**Data Preprocessing**:
- Feature Scaling: StandardScaler applied to normalize all features
- No missing values in the dataset
- All features are numeric and ready for modeling

---

## Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **K-Nearest Neighbor** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest** | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **XGBoost** | 0.9561 | 0.9907 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | **Best overall performance** with the highest accuracy (98.25%), AUC (99.54%), and MCC (0.9623). Despite being a simple linear model, it excels on this dataset, indicating that the classes are likely linearly separable after feature scaling. The balanced precision and recall (98.61%) show excellent performance on both classes, making it the most reliable model for this medical diagnosis task. |
| **Decision Tree** | **Lowest performance** among all models with accuracy of 91.23% and the lowest AUC (91.57%) and MCC (0.8174). The model shows signs of overfitting despite max_depth=10 constraint. The lower recall (90.28%) compared to precision (95.59%) suggests it's more conservative in predicting benign cases, which could lead to false negatives—a critical concern in medical diagnosis. |
| **K-Nearest Neighbor** | **Strong performance** with 95.61% accuracy and excellent AUC (97.88%). The model achieves balanced precision (95.89%) and recall (97.22%), indicating good generalization. The high recall is particularly valuable in medical contexts where missing a malignant case is more costly. Performance is sensitive to the choice of k (k=5 used here) and benefits significantly from feature scaling. |
| **Naive Bayes** | **Moderate performance** with 92.98% accuracy but an impressive AUC of 98.68%, second only to Logistic Regression. The model assumes feature independence, which may not fully hold for correlated medical features. However, it shows perfectly balanced precision and recall (94.44%), suggesting it doesn't favor either class. It's computationally efficient and works well despite its simplistic assumptions. |
| **Random Forest** | **Excellent ensemble performance** matching KNN in accuracy (95.61%) but with superior AUC (99.39%), the second-best overall. The ensemble of 100 trees provides robust predictions and handles non-linear relationships well. Identical precision and recall to KNN (95.89%, 97.22%) demonstrate consistent performance. The model is less prone to overfitting than single decision trees and provides valuable feature importance insights. |
| **XGBoost** | **Top-tier ensemble performance** with 95.61% accuracy and outstanding AUC (99.07%). The model achieves the **highest recall** (98.61%) among all models, meaning it's least likely to miss malignant cases—crucial for medical diagnosis where false negatives are dangerous. Slightly lower precision (94.67%) indicates more false positives, but this trade-off is acceptable in cancer detection. The gradient boosting approach effectively handles complex patterns in the data. |

---

## Key Insights

### Best Models for Medical Diagnosis:
1. **Logistic Regression** - Best overall metrics, simplest model
2. **Random Forest** - Excellent balance, provides interpretability through feature importance
3. **XGBoost** - Highest recall, best for minimizing false negatives

### Model Selection Considerations:
- **For Deployment**: Logistic Regression (simplest, highest accuracy, interpretable)
- **For Safety (minimize false negatives)**: XGBoost (highest recall)
- **For Interpretability**: Random Forest or Logistic Regression (feature importance analysis)

### Dataset Characteristics:
- The high performance across most models (>95% accuracy) suggests the dataset has well-separated classes with informative features
- Feature scaling significantly impacts performance, especially for distance-based (KNN) and gradient-based (Logistic Regression) models
- The dataset is slightly imbalanced (62.7% benign, 37.3% malignant) but not severely enough to require special handling

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Classification Script**:
   ```bash
   python ml_classification.py
   ```

3. **Output**:
   - Console output with detailed metrics for each model
   - `model_results.csv` - CSV file with all evaluation metrics
   - `breast_cancer_dataset.csv` - The dataset used for training

---

## File Structure

```
├── ml_classification.py          # Main script with all 6 ML models
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── breast_cancer_dataset.csv     # Dataset
└── model_results.csv             # Results table
```

---

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions (TP+TN)/(Total)
- **AUC (Area Under ROC Curve)**: Model's ability to discriminate between classes
- **Precision**: Of predicted positive cases, how many are actually positive (TP/(TP+FP))
- **Recall**: Of actual positive cases, how many were correctly identified (TP/(TP+FN))
- **F1 Score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix elements (-1 to +1)

---

## Conclusion

This comparative study demonstrates that for the Breast Cancer Wisconsin dataset, **Logistic Regression** provides the best overall performance despite its simplicity. However, in a medical context where missing a malignant case is critical, **XGBoost** with its highest recall (98.61%) might be the preferred choice. The ensemble methods (Random Forest and XGBoost) show robust performance, validating their reputation in medical machine learning applications.

---

## Author

**ML Classification Assignment**  
Date: February 2026
