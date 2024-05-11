from data.pemrosesan_data import ambilData, labelEncode, Split
from algoritma.knn import euclidean_distance
from sklearn.model_selection import train_test_split

# Main program
if __name__ == '__main__':

    # Load dataset
    dataset = ambilData() 
    
    # menampilkan dataset
    print ("\t\tHasil Import Dataset") 
    print (dataset)

    # encoding nilai kategorikal menjadi numerikal
    dataset = labelEncode(dataset) 
    
    # menampilkan dataset yang telah diencoding
    print ("\t\tDataset Yang Telah Diencoding")
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
    
    # Misalkan 'X' adalah matriks fitur dan 'y' adalah array target
    # Sesuaikan X dan y dengan dataset kamu
    X = dataset.drop('HeartDisease', axis=1)  # Drop kolom target jika ada
    y = dataset['HeartDisease']

    # Memisahkan dataset menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train)
    print(X_test)

    print(X.columns)
    print(X_test.iloc[0])
    ed = euclidean_distance(X_test.iloc[0], X_train.iloc[0], X)
    print(ed)

    
    