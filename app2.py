import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load trained model
model = joblib.load('model_prediction_fundraising.pkl')  

def main():
    st.title("Prediksi Fundraising App")

    # Input features from user
    age = st.slider("Usia", 18, 100, 25)
    income = st.slider("Pendapatan Bulanan", 4000000, 2000000, 500000)
    # Tambahkan fitur lain sesuai dengan model Anda

    # Predict button
    if st.button("Prediksi"):
        # Create a dataframe with the input features
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            # Tambahkan fitur lain sesuai dengan model Anda
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        st.success(f"Hasil Prediksi: {prediction[0]}")

if __name__ == "__main__":
    main()
