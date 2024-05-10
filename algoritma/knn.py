import pandas as pd
import numpy as np

def euclidean_distance(data_testing, data_training, X):
    """
    Menghitung jarak Euclidean antara dua baris dalam DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame yang berisi dataset.
        index_test (int): Indeks baris data testing.
        index_train (int): Indeks baris data training.

    Returns:
        float: Jarak Euclidean antara dua baris.
    """
    # Mengambil fitur yang ada dalam DataFrame
    features = X.columns
    
    # Menghitung jarak Euclidean
    distance = np.linalg.norm(data_testing[features] - data_training[features])
    
    return distance