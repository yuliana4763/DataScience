import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the pretrained model
model = joblib.load('C:/Users/ACER/SKRIPSI YULIANA APP/DataScience/model_prediction_fundraising1.pkl')

st.title("BAZNAS RI Digital Fundraising Prediction Application")

st.sidebar.title("Informasi Tambahan")
st.sidebar.write("Aplikasi ini memprediksi hasil fundraising menggunakan model yang telah dilatih dan berdasarkan input yang Anda berikan. Model yang digunakan dalam aplikasi ini adalah model yang telah di-train sebelumnya menggunakan data muzaki.")
st.sidebar.markdown("Hubungi kami di [yuliana111099@gmail.com](mailto:yuliana111099@gmail.com) untuk informasi lebih lanjut.")
st.sidebar.markdown("[Kunjungi situs web BAZNAS RI](https://baznas.go.id)")
st.sidebar.markdown("Jika ingin melakukan prediksi terhadap satu data, klik link berikut.")
st.sidebar.markdown("[Prediksi Satu Data](https://apppredict-ver2.streamlit.app/)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def preprocess_data(data):
    data = pd.read_csv(data)
    if "Unnamed: 0" in data.columns:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    return data

def predict(data, model):
    # Ensure only necessary columns are used
    X = data[["umur", "gender", "occupation_group", "periode_transaksi"]]
    X["gender"] = LabelEncoder().fit_transform(X["gender"])
    X["occupation_group"] = LabelEncoder().fit_transform(X["occupation_group"])
    X["periode_transaksi"] = LabelEncoder().fit_transform(X["periode_transaksi"])
    predictions = model.predict(X)
    return predictions

if uploaded_file is not None:
    data = preprocess_data(uploaded_file)
    predictions = predict(data, model)

    # Map numeric predictions to categories, assuming you have a mapping dictionary
    mapping = {0: 'Nominal Rendah', 3: 'Nominal Sedang', 2: 'Nominal Diatas Rata-rata', 1: 'Nominal Tinggi'}
    predictions_categorical = [mapping[pred] for pred in predictions]

    # Create a new DataFrame for display that does not include the 'nominal' and 'kategori_nominal' columns
    result_df = data.copy()
    result_df["Predictions"] = predictions_categorical

    # Display prediction results
    st.write(result_df[["umur", "gender", "occupation_group", "periode_transaksi", "Predictions"]])

    # Visualization
    st.subheader("Visualisasi Hasil Prediksi")
    prediction_counts = pd.Series(predictions_categorical).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    prediction_counts.plot(kind='bar', ax=ax, color='skyblue')
    plt.xlabel('Kategori Prediksi')
    plt.ylabel('Jumlah')
    plt.title('Distribusi Hasil Prediksi')
    plt.xticks(rotation=45)
    st.pyplot(fig)
