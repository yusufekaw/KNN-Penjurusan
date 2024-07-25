from data.pemrosesan_data import ambilData, labelEncode, Split
from algoritma.knn import euclidean_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from collections import Counter

# Main program
if __name__ == '__main__':

    # Load dataset
    dataset = ambilData() 
    
    # menampilkan dataset
    print ("\t\tHasil Import Dataset") 
    print (dataset)

    X_train, X_test, y_train, y_test = Split(dataset)

    print("X Train")
    print(X_train)
    print("X test")
    print(X_test)
    print("Y Train")
    print(y_train)
    print("Y test")
    print(y_test)

    peminat = dataset['Bidang_Minat'].value_counts()
    print(peminat)

    K = int(input('Masukkan K : '))
    
    matriks_jarak = []
    for i in range(len(X_test)):
        jarak = []
        for ii in range(len(X_train)):
            hitung_jarak = euclidean_distance(X_test.iloc[i],X_train.iloc[ii], X_train)
            jarak.append(hitung_jarak)
        matriks_jarak.append(jarak)

    # Menampilkan hasil
    df_jarak = pd.DataFrame(matriks_jarak, index=X_test.index, columns=X_train.index)
    print("Jarak Euclidean antara semua pasangan baris dalam dataset:")
    print(df_jarak)

    # Menemukan 3 jarak terdekat untuk setiap baris dalam data pengujian
    indeks_terdekat = df_jarak.apply(lambda row: row.nsmallest(K).index, axis=1)
    jarak_terdekat = df_jarak.apply(lambda row: row.nsmallest(K).values, axis=1)

    # Menampilkan hasil
    hasil_jarak = []
    for i, index_test in enumerate(X_test.index):
        k_indeks_terdekat = indeks_terdekat[index_test]
        k_jarak_tedekat = jarak_terdekat[index_test]
        for ii in range(K):
            index_train = k_indeks_terdekat[ii]
            jarak = k_jarak_tedekat[ii]
            kelas_train = y_train[index_train]
            hasil_jarak.append({
                'Index Test': index_test,
                'Index Train': index_train,
                'Jarak': jarak,
                'Kelas Train': kelas_train
            })

    hasil_jarak_df = pd.DataFrame(hasil_jarak)
    print("Jarak Terdekat dari Setiap Baris Data Pengujian ke Data Pelatihan:")
    print(hasil_jarak_df)

    # Menentukan kelas baru pada data pengujian berdasarkan kelas mayoritas dari tetangga terdekat
    kelas_prediksi = []
    for index_test in X_test.index:
        kelas_terdekat = hasil_jarak_df[hasil_jarak_df['Index Test'] == index_test]['Kelas Train']
        kelas_mayoritas = Counter(kelas_terdekat).most_common(1)[0][0]
        kelas_prediksi.append({
            'Index Test': index_test,
            'Kelas Prediksi':kelas_mayoritas
        })

    # Kelas prediksi
    kelas_prediksi_df = pd.DataFrame(kelas_prediksi)
    kelas_prediksi_df.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    hasil_prediksi = result = pd.concat([kelas_prediksi_df, y_test], axis=1)
    hasil_prediksi.rename(columns={'Bidang_Minat': 'Target'}, inplace=True)
    print("Kelas Baru pada Data Pengujian Berdasarkan Kelas Mayoritas dari Tetangga Terdekat:")
    print(hasil_prediksi)

    prediksi_benar = (hasil_prediksi['Kelas Prediksi'] == hasil_prediksi['Target']).sum()
    prediksi_salah = hasil_prediksi.shape[0]-prediksi_benar
    akurasi = (prediksi_benar/hasil_prediksi.shape[0])*100
    presisi = precision_score(hasil_prediksi['Kelas Prediksi'], hasil_prediksi['Target'], average='micro')*100
    recall = recall_score(hasil_prediksi['Kelas Prediksi'], hasil_prediksi['Target'], average='micro')*100
    f1 = f1_score(hasil_prediksi['Kelas Prediksi'], hasil_prediksi['Target'], average='micro')*100
    print("Prediksi Benar : ",prediksi_benar)
    print("Prediksi salah : ",prediksi_salah)
    print("Akurasi : ",round(akurasi,2),"%")
    print("Presisi : ",round(presisi,2),"%")
    print("Recall : ",round(recall,2),"%")
    print("F1 : ",round(f1,2),"%")

    # Jumlah peminat jurusan
    bidang_minat = pd.Series(list(set(dataset['Bidang_Minat']).union(set(kelas_prediksi_df['Kelas Prediksi']))))
    prediksi_peminat = kelas_prediksi_df['Kelas Prediksi'].value_counts().reindex(bidang_minat, fill_value=0)
    print(prediksi_peminat)