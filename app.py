import streamlit as st
import numpy as np
import pickle

st.title("Tugas Kelompok 6: Prediksi Risiko Kanker Menggunakan Machine Learning")

st.write("Silakan isi faktor faktor berikut untuk mendapatkan prediksi risiko kanker.")

# ===========================
# INPUT USER
# ===========================

age = st.number_input("Usia", min_value=1, max_value=120)

gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Pria" if x == 0 else "Wanita")

smoking = st.selectbox("Merokok", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

alcohol = st.selectbox("Konsumsi Alkohol", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

obesity = st.selectbox("Obesitas", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

family = st.selectbox("Riwayat Keluarga Kanker", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

red_meat = st.number_input("Konsumsi Daging Merah (gram per hari)", min_value=0)

processed_meat = st.number_input("Konsumsi Makanan Asin/Olahan (gram per hari)", min_value=0)

fruit_veg = st.number_input("Asupan Buah & Sayur (porsi per hari)", min_value=0)

physical_activity = st.number_input("Aktivitas Fisik per Minggu (menit)", min_value=0)

air_pollution = st.number_input("Tingkat Polusi Udara (0 - 10)", min_value=0, max_value=10)

occupational = st.selectbox(
    "Paparan Bahan Berbahaya di Tempat Kerja",
    [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya"
)

brca = st.selectbox("Mutasi Gen BRCA", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

hpylori = st.selectbox("Infeksi H. Pylori", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

calcium = st.number_input("Asupan Kalsium (mg per hari)", min_value=0)

bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)

pa_level = st.number_input("Level Aktivitas Fisik (1 - 5)", min_value=1, max_value=5)


# ===========================
# PREDIKSI
# ===========================

if st.button("Prediksi Risiko Kanker"):

    # ========== AUTO GENERATE SCORE ==========
    overall_score = (
        smoking * 20 +
        obesity * 15 +
        family * 25 +
        brca * 25 +
        alcohol * 10 +
        hpylori * 10 +
        occupational * 10 +
        air_pollution * 5 +
        (red_meat / 10) +
        (processed_meat / 10) +
        max(0, 5 - fruit_veg) * 3 +
        max(0, 30 - physical_activity / 10) +
        (bmi - 18) * 1.2
    )

    overall_score = min(max(int(overall_score), 0), 100)

    st.info(f"Skor Risiko Keseluruhan Anda: **{overall_score} / 100**")

    # ========== LOAD MODEL ==========
    with open("model_rf.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]

    # ========== SIAPKAN DATA ==========
    data = np.array([[
        age, gender, smoking, alcohol, obesity, family,
        red_meat, processed_meat, fruit_veg,
        physical_activity, air_pollution, occupational,
        brca, hpylori, calcium,
        overall_score, bmi, pa_level
    ]])

    data_scaled = scaler.transform(data)

    # ========== PREDIKSI ==========
    pred = model.predict(data_scaled)[0]

    st.subheader("Hasil Prediksi")

    if pred == 1:
        st.error("Risiko Kanker Tinggi")
    else:
        st.success("Risiko Kanker Rendah")
