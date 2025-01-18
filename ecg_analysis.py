import os
import wfdb
import pandas as pd
import numpy as np
from utils.utils import get_data
from utils import hrv
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# Ścieżka do folderu z rekordami
afdb_path = './afdb'

# Pobierz listę wszystkich plików w folderze afdb
records = [os.path.join(afdb_path, f.split('.')[0]) for f in os.listdir(afdb_path) if f.endswith('.hea')]

# Lista na wyniki
data = []

# Przetwarzanie każdego rekordu
for record_name in records:
    print(f"Processing record: {record_name}")

    # Wczytaj dane dla jednego odprowadzenia
    try:
        signal, properties = wfdb.rdsamp(record_name, channels=[0])  # channels=[0] dla pojedynczego odprowadzenia
        annotations = wfdb.rdann(record_name, 'atr')
        QRS = wfdb.rdann(record_name, 'qrs')
    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
        continue

    f = properties["fs"]  # Częstotliwość próbkowania
    T = 1 / f
    samples_per_minute = int(f * 60)

    # R-peaks i długość sygnału
    r_peaks = QRS.sample  # Lokalizacje R-peaków w próbkach
    signal_length = len(signal)  # Liczba próbek w sygnale
    num_segments = signal_length // samples_per_minute  # Liczba segmentów 1-minutowych

    # Przypisanie etykiety do R-peaków
    AnnotationRhythm = pd.Series(annotations.aux_note)
    AnnotationSamples = pd.Series(annotations.sample)

    labeled_Rpeaks = []
    num_labeled_Rpeaks = len(AnnotationSamples) - 1
    if num_labeled_Rpeaks == 0:
        continue

    for j in range(num_labeled_Rpeaks):
        df = pd.DataFrame(r_peaks[(r_peaks > AnnotationSamples[j]) & (r_peaks < AnnotationSamples[j + 1])])
        df['Label'] = AnnotationRhythm[j]
        labeled_Rpeaks.append(df)
    labeled_Rpeaks = pd.concat(labeled_Rpeaks)

    # Mapowanie etykiet na klasy (0 - Non-AF, 1 - AF)
    labeled_Rpeaks['Label'] = labeled_Rpeaks['Label'].map({'(N': 0, '(AFIB': 1})

    # Analiza segmentów
    for segment_idx in range(num_segments):
        start_sample = segment_idx * samples_per_minute
        end_sample = start_sample + samples_per_minute

        segment_r_peaks = r_peaks[(r_peaks >= start_sample) & (r_peaks < end_sample)]
        segment_labels = labeled_Rpeaks[(labeled_Rpeaks[0] >= start_sample) & (labeled_Rpeaks[0] < end_sample)][
            'Label']

        if segment_labels.nunique() > 1:  # Jeśli etykiety różnią się w ramach segmentu, pomiń ten segment
            continue

        # Przypisanie klasy 0 i 1
        segment_class = 1 if segment_labels.sum() > 0 else 0  # 1 dla AF, 0 dla Non-AF

        # Wyznaczanie RR-odstępy
        rr_intervals = [(segment_r_peaks[i + 1] - segment_r_peaks[i]) * T * 1000
                        for i in range(len(segment_r_peaks) - 1)]

        if rr_intervals:
            hrv_parameters = hrv.get_parameters(rr_intervals)
            hrv_parameters.append(segment_class)
            data.append(hrv_parameters)

# Przeksztańć dane na DataFrame
df = pd.DataFrame(data, columns=[
    'mean_rr', 'sdrr', 'sdsd', 'rmssd', 'median_rr', 'range_rr', 'cvsd', 'cvrr', 'mean_hr', 'max_hr', 'min_hr',
    'std_hr', 'Class'
])

# Podział na cechy (X) i etykiety (y)
X = df.drop('Class', axis=1)
y = df['Class']  # Używamy bezpośrednio klasy 0/1

# Normalizacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

    # Przechowywanie wyników
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'std_dev': std_dev  # Dodanie odchylenia standardowego do wyników
    }

    # Wyświetlanie wyników walidacji krzyżowej
    print(f"Cross-validation results for {model_name}: {cv_scores}")
    print(f"Mean accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Standard deviation of accuracy: {std_dev:.4f}")

# Porównanie wyników
results_df = pd.DataFrame(results).T  # Transponowanie, aby modele były wierszami
print("\nComparison of model performance (mean accuracy from cross-validation):")
print(results_df)
