# app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
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

if uploaded_file is not None:
    # Baca data dari file CSV
    data = pd.read_csv(uploaded_file)

    # Tampilkan struktur data
    st.subheader("Struktur Data yang Diunggah")
    st.write(data.head())

    # Tampilkan statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())

    # Lakukan prediksi
    predictions = model.predict(data)

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(predictions)

     # Visualisasi hasil prediksi
    st.subheader("Visualisasi Hasil Prediksi")
    plt.figure(figsize=(8, 5))
    sns.countplot(predictions, palette="viridis")
    plt.title("Distribusi Hasil Prediksi")
    st.pyplot()

    # Tampilkan confusion matrix dan classification report jika ada kolom target
    if 'kategori_nominal' in data.columns:
        X_train, X_test, y_train, y_test = train_test_split(data.drop('kategori_nominal', axis=1), data['target_column'], test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot()

        st.subheader("Classification Report")
        cr = classification_report(y_test, y_pred)
        st.text(cr)

   