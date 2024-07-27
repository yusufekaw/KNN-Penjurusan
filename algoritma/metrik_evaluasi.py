import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#metrik evaluasi
def metrik(prediksi, target):
    #label
    labels = sorted(target.unique().tolist() + prediksi.unique().tolist())
    labels = sorted(set(labels))
    
    # Buat confusion matrix
    cm = confusion_matrix(target, prediksi, labels=labels)
    # Buat DataFrame untuk confusion matrix
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    #prediksi benar
    T = (prediksi == target).sum()
    #prediksi salah
    F = prediksi.shape[0] - T
    #akurasi, presisi, recall, f1
    akurasi = accuracy_score(prediksi, target)
    presisi = precision_score(prediksi, target, average='macro')
    recall = recall_score(prediksi, target, average='macro')
    f1 = f1_score(prediksi, target, average='macro')
    return cm, T, F, akurasi, presisi, recall, f1


def visualisasiCM(cm):
    # Visualisasi confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')
    plt.show()

