# app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load model yang telah dilatih
model = joblib.load('model_prediction_fundraising.pkl')

# Implementasi streamlit untuk antarmuka pengguna
st.title("Fundraising Digital Prediction Web App BAZNAS RI")

# Sidebar dengan informasi tambahan
st.sidebar.title("Informasi Tambahan")
st.sidebar.write("Aplikasi ini memprediksi hasil fundraising menggunakan model yang telah dilatih dan berdasarkan input yang Anda berikan. Model yang digunakan dalam aplikasi ini adalah model yang telah di-train sebelumnya menggunakan data muzaki.")
st.sidebar.write("Hubungi kami di yuliana111099@gmail.com untuk informasi lebih lanjut.")

# Upload file CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def preprocessing_data(data):
    data = pd.read_csv(data)
    data.drop(["Unnamed: 0"], axis=1, inplace=True)
    
    return data

def predict(data, model):
    X = data[["umur", "gender", "periode_transaksi"]]
    nominal = LabelEncoder().fit(data["kategori_nominal"])
    predicts = model.predict(X)
    labels = nominal.inverse_transform(predicts)
    return predicts, labels

def split(data):
    nominal = LabelEncoder().fit(data["kategori_nominal"])
    data["kategori_nominal"] = nominal.transform(data["kategori_nominal"])
    
    X = data[["umur", "gender", "periode_transaksi"]]
    y = data["kategori_nominal"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


if uploaded_file is not None:
    # Baca data dari file CSV
    data = preprocessing_data(uploaded_file)
    
    # Tampilkan struktur data
    st.subheader("Struktur Data yang Diunggah")
    st.write(data.head())

    # Tampilkan statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())

    # Lakukan prediksi
    predictions, labels = predict(data, model)

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(labels)

     # Visualisasi hasil prediksi
    st.subheader("Visualisasi Hasil Prediksi")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(labels, palette="viridis", ax=ax)
    plt.title("Distribusi Hasil Prediksi")
    st.pyplot(fig)

    # Tampilkan confusion matrix dan classification report jika ada kolom target
    if 'kategori_nominal' in data.columns:
        X_train, X_test, y_train, y_test = split(data)
        y_pred = model.predict(X_test)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        cr = classification_report(y_test, y_pred)
        st.text(cr)

   