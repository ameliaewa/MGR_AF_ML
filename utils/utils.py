import os

import numpy as np
import pandas as pd
import wfdb


def get_data(db_name):
    if os.path.isdir(f"./{db_name}"):
        print('You already have the data.')
    else:
        wfdb.dl_database(f'{db_name}', db_name)


def get_first_from_fragment(df: pd.DataFrame):
    """
    Function for determining where new fragments (N or AFIG) are beggining in the array.
    :param df: dataframe containing records of R peaks with label
    :return: tuple of lists with first elements of fragments
    """
    negative_starts = []
    positive_starts = []
    in_label = False
    for index, row in df.iterrows():
        if row['Label'] == 0 and not in_label:
            negative_starts.append(row['Rpeaks'])
            in_label = True
        elif row['Label'] != 0 and in_label:
            positive_starts.append(row['Rpeaks'])
            in_label = False
    return negative_starts, positive_starts


def get_rr_interval_for_period(r_peaks, starts, pidx, f):
    """
    Function for getting rr intervals for a period of time from a set of peaks.
    :param r_peaks: list of sample number of R peaks
    :param starts: list of beggining of the peroids
    :param pidx: peroid index
    :param f: sampling frequency
    :return: list of rr intervals
    """
    samples_per_minutes = 60*f
    T=1/f
    return [(r_peaks[i + 1] - r_peaks[i]) * T * 1000 for i in
                             range(np.where(r_peaks > starts[pidx])[0][0],
                                   np.where(r_peaks < starts[pidx] + samples_per_minutes)[0][-1])]
