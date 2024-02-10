# app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load model yang telah dilatih
model = joblib.load('model_prediction_fundraising.pkl')

# Implementasi streamlit untuk antarmuka pengguna
st.title("BAZNAS RI Digital Fundraising Prediction Application")

# Sidebar dengan informasi tambahan
st.sidebar.title("Informasi Tambahan")
st.sidebar.write("Aplikasi ini memprediksi hasil fundraising menggunakan model yang telah dilatih dan berdasarkan input yang Anda berikan. Model yang digunakan dalam aplikasi ini adalah model yang telah di-train sebelumnya menggunakan data muzaki.")
st.sidebar.markdown("Hubungi kami di [yuliana111099@gmail.com](mailto:yuliana111099@gmail.com) untuk informasi lebih lanjut.")
st.sidebar.markdown("[Kunjungi situs web BAZNAS RI](https://baznas.go.id)")
st.sidebar.markdown("Jika ingin melakukan prediksi terhadap satu data, klik link berikut.")
st.sidebar.markdown("[Prediksi Satu Data](https://www.linkedin.com/in/yuliana-35739a16a/)")

# Upload file CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def preprocessing_data(data):
    data = pd.read_csv(data)
    data.drop(["Unnamed: 0"], axis=1, inplace=True)
    
    return data

def predict(data, model):
    X = data[["umur", "gender", "occupation_group", "periode_transaksi"]]
    X["occupation_group"] = LabelEncoder().fit_transform(X["occupation_group"])
    nominal = LabelEncoder().fit(data["kategori_nominal"])
    predicts = model.predict(X)
    labels = nominal.inverse_transform(predicts)
    return predicts, labels

def split(data):
    nominal = LabelEncoder().fit(data["kategori_nominal"])
    data["kategori_nominal"] = nominal.transform(data["kategori_nominal"])
    data["occupation_group"] = LabelEncoder().fit_transform(data["occupation_group"])
    data.drop(["kategori_nominal"], axis=1, inplace=True)

    X = data[["umur", "gender", "occupation_group", "periode_transaksi"]]
    y = data["kategori_nominal"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if uploaded_file is not None:
    # Baca data dari file CSV
    data = preprocessing_data(uploaded_file)

    # Lakukan prediksi
    predictions, labels = predict(data, model)
    
    label_encoder = LabelEncoder()
    predictions_numeric = label_encoder.fit_transform(predictions)

    # Mapping nilai numerik ke nilai kategorikal yang diinginkan
    mapping = {0: 'Nominal Rendah', 3: 'Nominal Sedang', 2: 'Nominal Diatas Rata-rata', 1: 'Nominal Tinggi'}

    # Fungsi untuk memetakan nilai numerik ke nilai kategorikal
    def map_to_categorical(label):
         return mapping[label]

    # Terapkan fungsi map_to_categorical pada kolom predictions
    predictions_categorical = list(map(map_to_categorical, predictions_numeric))

    # Hasil prediksi
    labels = np.array([0])  
    
    data_df = data.copy()  # Salin DataFrame untuk mempertahankan data asli
    data_df["predictions"] = predictions

    # Menambahkan kolom baru ke dalam DataFrame hasil prediksi
    data_df['predictions'] = predictions_categorical

    st.write(data_df)

    # Visualisasi hasil prediksi
    st.subheader("Visualisasi Hasil Prediksi")

    # Menghitung frekuensi masing-masing kategori
    prediction_counts = data_df['predictions'].value_counts()

    # Membuat plot bar untuk menampilkan distribusi hasil prediksi
    fig, ax = plt.subplots(figsize=(8, 5))
    prediction_counts.plot(kind='bar', ax=ax, color='skyblue')
    plt.xlabel('Kategori Prediksi')
    plt.ylabel('Jumlah')
    plt.title('Distribusi Hasil Prediksi')
    plt.xticks(rotation=45)
    st.pyplot(fig)

