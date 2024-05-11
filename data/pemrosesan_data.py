import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#fungsi untuk memuat dataset
def ambilData():
    # Path ke file dataset
    #path = "dataset/heart.csv"
    # Membaca dataset menggunakan Pandas
    dataset = pd.read_csv("data/dataset/heart.csv")
    return dataset

#Label Encoding mengubah nilai kategorikal menjadi numerikal
def labelEncode(dataset):
    le = LabelEncoder()
    for kolom in dataset.columns:
        if dataset[kolom].dtype == 'object': #mengambil kolom bertipe data objek
            dataset[kolom] = le.fit_transform(dataset[kolom]) #mengganti nilai kategori menjadi angka
    return dataset

#split dataset (training testing)
def Split(dataset):
    n = int(len(dataset) * 0.7) #maks index dataset

    # Memisahkan fitur dan target
    X = dataset.drop('HeartDisease', axis=1)  # Drop kolom target jika ada
    y = dataset['HeartDisease']

     # Memisahkan dataset berdasarkan proporsi
    X_train = X.iloc[:n]
    X_test = X.iloc[n:]
    y_train = y.iloc[:n]
    y_test = y.iloc[n:]

    return X_train, X_test, y_train, y_test