import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("model_prediction_fundraising.pkl")

# Fungsi prediksi
def predict(data):
    prediction = model.predict(data)
    return prediction

# Tampilan aplikasi
def main():
    st.title("BAZNAS RI Digital Fundraising Prediction Application v.2")

    # Input data untuk prediksi
    umur = st.slider("Umur", min_value=18, max_value=80, value=30)
    gender = st.radio("Gender", ["pria", "wanita"])
    occupation_group = st.selectbox("Profesi", ["Amil", "PNS", "Lainnya", "Tidak Bekerja", "Wiraswasta",
                                                         "Pegawai Swasta", "Konsultan", "Industri", "Staff", "IT",
                                                         "Akuntansi", "Kesehatan", "Energi", "Marketing", "Teknisi",
                                                         "Auditor", "Freelancer", "Transportasi", "TNI/POLRI", "Dokter"])
    periode_transaksi = st.selectbox("Periode Transaksi", ["Reguler", "Ramadhan", "Kurban"])
    kategori_nominal = st.selectbox("Kategori Nominal", ["Nominal Tinggi", "Nominal Diatas Rata-rata", "Nominal Sedang", "Nominal Rendah"])
    
    # Prediksi
    if st.button("Prediksi Fundraising"):
        # Membuat array 2D dari input
        data = np.array([[umur, gender, occupation_group, periode_transaksi, kategori_nominal]])

        # Label encoding untuk kategori
        le_gender = LabelEncoder().fit(['pria', 'wanita'])
        le_occupation_group = LabelEncoder().fit(["Amil", "PNS", "Lainnya", "Tidak Bekerja", "Wiraswasta",
                                                 "Pegawai Swasta", "Konsultan", "Industri", "Staff", "IT",
                                                 "Akuntansi", "Kesehatan", "Energi", "Marketing", "Teknisi",
                                                 "Auditor", "Freelancer", "Transportasi", "TNI/POLRI", "Dokter"])
        le_periode_transaksi = LabelEncoder().fit(["Reguler", "Ramadhan", "Kurban"])
        le_kategori_nominal = LabelEncoder().fit(["Nominal Tinggi", "Nominal Diatas Rata-rata", "Nominal Sedang", "Nominal Rendah"])

        # Transformasi input menjadi array untuk prediksi
        input_data[:, 1] = le_gender.transform(input_data[:, 1])
        input_data[:, 2] = le_occupation_group.transform(input_data[:, 2])
        input_data[:, 3] = le_periode_transaksi.transform(input_data[:, 3])
        input_data[:, 4] = le_kategori_nominal.transform(input_data[:, 4])
        

        # Prediksi dengan model
        result = predict(data)

        # Mapping nilai numerik ke nilai kategorikal yang diinginkan
        mapping = {0: 'Nominal Rendah', 3: 'Nominal Sedang', 2: 'Nominal Diatas Rata-rata', 1: 'Nominal Tinggi'}
        result_categorical = mapping[result[0]]

        st.success(f"Hasil Prediksi Fundraising: {result_categorical}")

if __name__ == "__main__":
    main()
