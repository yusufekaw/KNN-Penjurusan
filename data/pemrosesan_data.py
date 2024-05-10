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