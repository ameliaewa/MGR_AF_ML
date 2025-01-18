import wfdb
import pandas as pd
from MGR_AF_ML.utils.hrv import  get_parameters


def label_r_peaks(record):
    # Wczytaj dane dla jednego odprowadzenia
    try:
        signal, properties = wfdb.rdsamp(record, channels=[0])  # channels=[0] dla pojedynczego odprowadzenia
        annotations = wfdb.rdann(record, 'atr')
        QRS = wfdb.rdann(record, 'qrs')
    except Exception:
        return

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

    labeled_r_peaks = []
    num_labeled_r_peaks = len(AnnotationSamples) - 1
    if num_labeled_r_peaks == 0:
        return

    for j in range(num_labeled_r_peaks):
        df = pd.DataFrame(r_peaks[(r_peaks > AnnotationSamples[j]) & (r_peaks < AnnotationSamples[j + 1])])
        df['Label'] = AnnotationRhythm[j]
        labeled_r_peaks.append(df)
    labeled_r_peaks = pd.concat(labeled_r_peaks)

    # Mapowanie etykiet na klasy (0 - Non-AF, 1 - AF)
    labeled_r_peaks['Label'] = labeled_r_peaks['Label'].map({'(N': 0, '(AFIB': 1})

    return labeled_r_peaks, r_peaks, num_segments, samples_per_minute, T


def segment_and_label(segment_idx, labeled_r_peaks, r_peaks, samples_per_minute):
    start_sample = segment_idx * samples_per_minute
    end_sample = start_sample + samples_per_minute
    segment_r_peaks = r_peaks[(r_peaks >= start_sample) & (r_peaks < end_sample)]
    segment_labels = labeled_r_peaks[(labeled_r_peaks[0] >= start_sample) & (labeled_r_peaks[0] < end_sample)][
        'Label']
    return segment_r_peaks, segment_labels


def process_ecg_records(records):
    data = []

    # Przetwarzanie każdego rekordu
    for record in records:

        print(f"Processing record: {record}")

        try:
            labeled_r_peaks, r_peaks, num_segments, samples_per_minute, T = label_r_peaks(record)
        except Exception as e:
            print(f"Error processing record {record}: {e}")
            continue

        # Analiza segmentów
        for segment_idx in range(num_segments):
            segment_r_peaks, segment_labels = segment_and_label(segment_idx, labeled_r_peaks, r_peaks, samples_per_minute)

            # Jeśli etykiety różnią się w ramach segmentu, pomiń ten segment
            if segment_labels.nunique() > 1:
                continue

            # Przypisanie klasy 0 i 1
            segment_class = 1 if segment_labels.sum() > 0 else 0  # 1 dla AF, 0 dla Non-AF

            # Wyznaczanie RR-odstępy
            rr_intervals = [(segment_r_peaks[i + 1] - segment_r_peaks[i]) * T * 1000
                            for i in range(len(segment_r_peaks) - 1)]

            if rr_intervals:
                hrv_parameters = get_parameters(rr_intervals)
                hrv_parameters.append(segment_class)
                data.append(hrv_parameters)

    # Przeksztańć dane na DataFrame
    df = pd.DataFrame(data, columns=[
        'mean_rr', 'sdrr', 'sdsd', 'rmssd', 'median_rr', 'range_rr', 'cvsd', 'cvrr', 'mean_hr', 'max_hr', 'min_hr',
        'std_hr', 'Class'
    ])
    return df
