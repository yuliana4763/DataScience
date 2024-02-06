import streamlit as st
import pandas as pd
import numpy  as nm
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Load model
model = joblib.load("model_prediction_fundraising.pkl")

# Fungsi prediksi
def predict(data):
    # st.write(data)
    prediction = model.predict(data)
    return prediction

# Tampilan aplikasi
def main():
    st.title("Aplikasi Prediksi Fundraising")

    # Input data untuk prediksi
    umur = st.slider("umur", min_value=18, max_value=80, value=30)
    gender = st.radio("gender", ["pria", "wanita"])
    occupation_group = st.selectbox("occupation_group", ["Amil", "PNS", "Lainnya", "Tidak Bekerja", "Wiraswasta",
                                                         "Pegawai Swasta", "Konsultan", "Industri", "Staff", "IT",
                                                         "Akuntansi", "Kesehatan", "Energi", "Marketing", "Teknisi",
                                                         "Auditor", "Freelancer", "Transportasi", "TNI/POLRI", "Dokter"])
    nominal = st.number_input("nominal", min_value=0, value=1000)
    periode_transaksi = st.selectbox("periode_transaksi", ["Reguler", "Ramadhan", "Kurban"])
    #data = umur+geder
    # Prediksi
    if st.button("Prediksi Fundraising"):
        #data = nm.array([[umur,gender,occupation_group,periode_transaksi]])
        data = [umur,gender,occupation_group,periode_transaksi]
        result = predict(data)
        st.success(f"Hasil Prediksi Fundraising: {result[0]}")


def predictx(data, model):
    #X = data[["umur", "gender", "occupation_group", "periode_transaksi"]]
    X = data
    # st.write(X[0][2])
   # X["occupation_group"] = LabelEncoder().fit_transform(X["occupation_group"])
    nominal = LabelEncoder().fit([X[0][0]])
    st.write(nominal)
    X= LabelEncoder().fit_transform(X[0])
   # nominal = LabelEncoder().fit(data["kategori_nominal"])
    # st.write(X[0][0])
    
    predicts = model.predict(X)
    labels = nominal.inverse_transform(predicts)
    #return predicts, labels
    return predicts

def split(data):
    nominal = LabelEncoder().fit(data["kategori_nominal"])
    data["kategori_nominal"] = nominal.transform(data["kategori_nominal"])
    data["occupation_group"] = LabelEncoder().fit_transform(data["occupation_group"])

    X = data[["umur", "gender", "occupation_group", "periode_transaksi"]]
    y = data["kategori_nominal"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    main()
