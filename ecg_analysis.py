import os
import pandas as pd
import numpy as np

from MGR_AF_ML.utils.ecg_processing import process_ecg_records
from utils.utils import get_data
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.ensemble import StackingClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# Definiowanie modeli
models = {
    'AdaBoost': AdaBoostClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'Stacking': StackingClassifier(
        estimators=[('rf', RandomForestClassifier()), ('svm', SVC())], final_estimator=RandomForestClassifier()
    ),
    # 'DNN': Sequential([
    #     Dense(64, input_dim=X_train.shape[1], activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
}


# Ścieżka do folderu z rekordami
afdb_path = './afdb'

# Pobierz listę wszystkich plików w folderze afdb
records = [os.path.join(afdb_path, f.split('.')[0]) for f in os.listdir(afdb_path) if f.endswith('.hea')]

# Define a list of segment lengths to test
segment_lengths = [10, 20, 30, 60, 120]

# Dictionary to store results for each segment length
experiment_results = {}

for num_of_seconds in segment_lengths:
    print(f"\nProcessing ECG records with segment length: {num_of_seconds} seconds")

    # Process ECG records with the given segment length
    df = process_ecg_records(records, num_of_seconds)

    # Split features (X) and labels (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    if X.isnull().sum().sum() > 0:
        print("Found missing values. Handling them...")
        # Option 1: Drop rows with missing values
        X = X.dropna()
        y = y[X.index]  # Alig

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize results for this segment length
    segment_results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} for segment length {num_of_seconds} seconds using cross-validation...")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        # Cross-validation predictions
        y_pred = cross_val_predict(model, X_scaled, y, cv=5)

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred, labels=np.unique(y))
        tp = np.diag(cm)  # True Positives
        fp = cm.sum(axis=0) - tp  # False Positives
        fn = cm.sum(axis=1) - tp  # False Negatives
        tn = cm.sum() - (fp + fn + tp)  # True Negatives

        # Calculate sensitivity and specificity for each class
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Weighted averages for sensitivity and specificity
        sensitivity_weighted = np.average(sensitivity, weights=np.bincount(y))
        specificity_weighted = np.average(specificity, weights=np.bincount(y))

        # Calculate other metrics
        accuracy = np.mean(cv_scores)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        std_dev = np.std(cv_scores)

        # Print metrics
        print(f"Cross-validation results for {model_name} (segment length {num_of_seconds}): {cv_scores}")
        print(f"Mean accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-score (weighted): {f1:.4f}")
        print(f"Sensitivity (weighted): {sensitivity_weighted:.4f}")
        print(f"Specificity (weighted): {specificity_weighted:.4f}")
        print(f"Standard deviation of accuracy: {std_dev:.4f}")

        # Store results
        segment_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity_weighted,
            'specificity': specificity_weighted,
            'std_dev': std_dev
        }

    # Save results for this segment length
    experiment_results[num_of_seconds] = segment_results


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Summarize and compare results
summary_df = pd.concat({seg: pd.DataFrame(res).T for seg, res in experiment_results.items()}, axis=0)
summary_df.index.names = ['Segment Length (s)', 'Model']
print("\nSummary of experiments:")
print(summary_df)
