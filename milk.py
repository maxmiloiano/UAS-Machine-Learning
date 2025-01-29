import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

# Judul aplikasi
st.title("Water Quality Prediction App")

# Input parameter dari user
st.sidebar.header("Masukkan Parameter")

def user_input():
    pH = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    Temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, value=25.0, step=0.1)
    Taste = st.sidebar.selectbox("Taste (0=Tidak Layak, 1=Layak)", [0, 1])
    Odor = st.sidebar.selectbox("Odor (0=Tidak Layak, 1=Layak)", [0, 1])
    Fat = st.sidebar.number_input("Fat", min_value=0.0, value=1.0, step=0.1)
    Turbidity = st.sidebar.number_input("Turbidity", min_value=0.0, value=1.0, step=0.1)
    Colour = st.sidebar.slider("Colour (240-255)", min_value=240, max_value=255, value=250)

    return {
        "pH": pH,
        "Temperature": Temperature,
        "Taste": Taste,
        "Odor": Odor,
        "Fat": Fat,
        "Turbidity": Turbidity,
        "Colour": Colour
    }

input_data = user_input()

# Jika Taste = 0 dan Odor = 0, tampilkan hasil langsung tanpa prediksi model
if input_data["Taste"] == 0 and input_data["Odor"] == 0:
    st.subheader("Hasil Prediksi:")
    st.write("Logistic Regression: Tidak Layak")
    st.write("Random Forest: Tidak Layak")
else:
    # Cek apakah file tersedia sebelum meminta unggahan
    file_path = "processed_data.csv"
    if not os.path.exists(file_path):
        uploaded_file = st.file_uploader("Upload processed_data.csv", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.error("File 'processed_data.csv' tidak ditemukan. Silakan upload file untuk melanjutkan.")
            st.stop()
    else:
        data = pd.read_csv(file_path)

    # Pisahkan fitur dan target pada data
    X = data.drop('Grade', axis=1)
    y = data['Grade']

    # Buat dataframe dari input user, memastikan urutan dan nama kolom sama dengan X
    new_data = pd.DataFrame([input_data])

    # Pastikan fitur input user sesuai skala data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    new_data_scaled = scaler.transform(new_data)

    # Inisialisasi model
    logreg_model = LogisticRegression(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)

    # Latih model dengan data yang sudah ada
    logreg_model.fit(X_scaled, y)
    rf_model.fit(X_scaled, y)

    # Prediksi dengan data baru dari input user
    logreg_pred = logreg_model.predict(new_data_scaled)
    rf_pred = rf_model.predict(new_data_scaled)

    # Menampilkan hasil prediksi
    logreg_result = 'Layak' if logreg_pred[0] == 1 else 'Tidak Layak'
    rf_result = 'Layak' if rf_pred[0] == 1 else 'Tidak Layak'

    st.subheader("Hasil Prediksi:")
    st.write(f"Logistic Regression: {logreg_result}")
    st.write(f"Random Forest: {rf_result}")
