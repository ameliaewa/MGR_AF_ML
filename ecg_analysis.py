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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
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

df = process_ecg_records(records,30)

# Podział na cechy (X) i etykiety (y)
X = df.drop('Class', axis=1)
y = df['Class']  # Używamy bezpośrednio klasy 0/1

# Normalizacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Słownik na wyniki
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name} using cross-validation...")

    # Przeprowadzanie krzyżowej walidacji (5-fold cross-validation)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # Obliczanie przewidywań dla całego zbioru danych przy użyciu krzyżowej walidacji
    y_pred = cross_val_predict(model, X_scaled, y, cv=5)

    # Obliczanie metryk
    accuracy = np.mean(cv_scores)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    std_dev = np.std(cv_scores)  # Obliczanie odchylenia standardowego z wyników krzyżowej walidacji



    # Wyświetlanie wyników walidacji krzyżowej
    print(f"Cross-validation results for {model_name}: {cv_scores}")
    print(f"Mean accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Standard deviation of accuracy: {std_dev:.4f}")

    # Przechowywanie wyników
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'std_dev': std_dev  # Dodanie odchylenia standardowego do wyników
    }

# Porównanie wyników
results_df = pd.DataFrame(results).T  # Transponowanie, aby modele były wierszami
print("\nComparison of model performance (mean accuracy from cross-validation):")
print(results_df)
