import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="Prediksi Drop Out Mahasiswa", layout="centered")
st.title("Dashboard Prediksi Drop Out / Lulus Mahasiswa Menggunakan Naive Bayes")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_kelulusan_mahasiswa.csv")
    df['Jumlah Semester'] = df['Jumlah Semester'].clip(upper=14)
    df['Pekerjaan Sambil Kuliah'] = df['Pekerjaan Sambil Kuliah'].map({'Ya': 1, 'Tidak': 0})
    df['Status Kelulusan'] = df['Status Kelulusan'].map({'Lulus': 0, 'Drop Out': 1})
    return df

df = load_data()

# --- FITUR & TARGET ---
features = ['IPK', 'Mata Kuliah Tidak Lulus', 'Jumlah Cuti Akademik',
            'IPS Rata-rata', 'Pekerjaan Sambil Kuliah', 'Jumlah Semester', 'IPS Tren']
features = [f for f in features if f in df.columns]
X = df[features]
y = df['Status Kelulusan']

# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAIN MODEL ---
model = GaussianNB()
model.fit(X_train, y_train)

# --- FORM INPUT PREDIKSI ---
st.header("Masukkan Data Mahasiswa untuk Prediksi")
with st.form("prediksi_form"):
    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=2.75, step=0.01)
    mknl = st.number_input("Mata Kuliah Tidak Lulus", min_value=0, max_value=12, value=1)
    cuti = st.number_input("Jumlah Cuti Akademik", min_value=0, max_value=2, value=0)
    ips_rata = st.number_input("IPS Rata-rata", min_value=0.0, max_value=4.0, value=2.75, step=0.01)
    kerja = st.selectbox("Pekerjaan Sambil Kuliah", options=["Tidak", "Ya"])
    kerja_enc = 1 if kerja == "Ya" else 0
    jml_semester = st.number_input("Jumlah Semester", min_value=1, max_value=14, value=8)

    if 'IPS Tren' in features:
        ips_tren = st.number_input("IPS Tren", value=0.0, step=0.01)
        data_pred = np.array([[ipk, mknl, cuti, ips_rata, kerja_enc, jml_semester, ips_tren]])
    else:
        data_pred = np.array([[ipk, mknl, cuti, ips_rata, kerja_enc, jml_semester]])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    hasil = model.predict(data_pred)[0]
    proba = model.predict_proba(data_pred)[0][hasil]
    st.subheader("Hasil Prediksi")
    if hasil == 0:
        st.success(f"Mahasiswa DIPREDIKSI **LULUS** (Probabilitas: {proba:.2f}) 🎓")
    else:
        st.error(f"Mahasiswa DIPREDIKSI **DROP OUT** (Probabilitas: {proba:.2f}) 💥")

st.divider()
st.header("Evaluasi Model Naive Bayes (Data Uji)")

# Akurasi & ROC
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
accuracy = (y_pred == y_test).mean() * 100
roc_auc = roc_auc_score(y_test, y_proba) * 100

col1, col2 = st.columns(2)
col1.metric("Akurasi Uji", f"{accuracy:.2f} %")
col2.metric("ROC-AUC Uji", f"{roc_auc:.2f} %")

# Classification report
with st.expander("Lihat Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Confusion matrix
with st.expander("Lihat Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig_cm)

# ROC Curve
with st.expander("Lihat ROC Curve"):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}%)')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Naive Bayes')
    ax.legend()
    st.pyplot(fig_roc)
