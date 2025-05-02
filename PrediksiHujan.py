import streamlit as st
import numpy as np
import tensorflow as tf
import os

# Konfigurasi halaman harus menjadi perintah Streamlit pertama
st.set_page_config(page_title="Prediksi Hujan", layout="centered")

st.title("🌧️ Prediksi Hujan Berdasarkan Cuaca")
st.markdown("Masukkan parameter cuaca untuk memprediksi apakah akan terjadi hujan atau tidak.")

# Tentukan path file model dengan benar
model_path = 'model.h5'

# Periksa apakah file model ada
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        model_loaded = True
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        model_loaded = False
else:
    st.error(f"File model tidak ditemukan di lokasi: {os.path.abspath(model_path)}")
    model_loaded = False

# Form untuk input data
day = st.number_input("Hari (1-31)", min_value=1, max_value=31, value=15)
pressure = st.number_input("Tekanan Udara (hPa)", value=1013.25)
maxtemp = st.number_input("Suhu Maksimum (°C)", value=30.0)
temperature = st.number_input("Suhu Rata-rata (°C)", value=27.0)  # Perbaikan typo
mintemp = st.number_input("Suhu Minimum (°C)", value=24.0)
dewpoint = st.number_input("Titik Embun (°C)", value=22.0)
humidity = st.number_input("Kelembaban (%)", min_value=0, max_value=100, value=80)
cloud = st.number_input("Tingkat Awan (%)", min_value=0, max_value=100, value=50)
sunshine = st.number_input("Jam Sinar Matahari", min_value=0.0, value=6.0)
winddirection = st.number_input("Arah Angin (derajat)", min_value=0.0, max_value=360.0, value=180.0)
windspeed = st.number_input("Kecepatan Angin (km/jam)", value=10.0)

if st.button("Prediksi"):
    if not model_loaded:
        st.error("Tidak dapat melakukan prediksi karena model belum dimuat.")
    else:
        try:
            # Menyiapkan data input untuk prediksi
            input_data = np.array([[day, pressure, maxtemp, temperature, mintemp,
                                    dewpoint, humidity, cloud, sunshine, winddirection, windspeed]])

            # Melakukan prediksi
            prediction = model.predict(input_data)

            # Interpretasi hasil prediksi
            if prediction.shape[1] == 1:
                pred_class = int(prediction[0][0] > 0.5)
                confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
            else:
                pred_class = np.argmax(prediction)
                confidence = np.max(prediction)

            # Menampilkan hasil
            st.subheader("📊 Hasil Prediksi")
            st.write(f"**Akurasi Prediksi (Confidence):** {confidence * 100:.2f}%")

            if pred_class == 1:
                st.success("🌧️ **Prediksi: Akan HUJAN.**")
            else:
                st.info("☀️ **Prediksi: Tidak Hujan.**")
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {str(e)}")
