import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Prediksi Drop Out Mahasiswa", layout="centered")
st.title("Dashboard Prediksi Drop Out / Lulus Mahasiswa Menggunakan Naive Bayes")

# --- SIDEBAR NAVIGASI ---
menu = st.sidebar.radio(
    "Navigasi",
    ("Prediksi & Evaluasi", "Clustering")
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_kelulusan_mahasiswa.csv")
    # Outlier removal pada 'Mata Kuliah Tidak Lulus'
    Q1 = df['Mata Kuliah Tidak Lulus'].quantile(0.25)
    Q3 = df['Mata Kuliah Tidak Lulus'].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    df = df[(df['Mata Kuliah Tidak Lulus'] >= batas_bawah) & (df['Mata Kuliah Tidak Lulus'] <= batas_atas)]
    # Encoding
    df['Pekerjaan Sambil Kuliah'] = df['Pekerjaan Sambil Kuliah'].map({'Tidak': 0, 'Ya': 1})
    df['Kategori Kehadiran'] = df['Kategori Kehadiran'].map({'Rendah': 0, 'Sedang': 1, 'Tinggi': 2})
    df['Status Kelulusan'] = df['Status Kelulusan'].map({'Drop Out': 0, 'Lulus': 1})
    df['Jumlah Semester'] = df['Jumlah Semester'].clip(upper=14)
    return df

df = load_data()

if menu == "Prediksi & Evaluasi":
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
        if hasil == 1:
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

elif menu == "Clustering":
    st.header("Clustering Mahasiswa (KMeans)")

    # Fitur clustering sesuai notebook (tanpa target & IPS Semester Akhir)
    fitur_cluster = [
        'IPK', 'Mata Kuliah Tidak Lulus', 'Jumlah Cuti Akademik',
        'Pekerjaan Sambil Kuliah', 'Jumlah Semester', 'IPS Rata-rata', 'IPS Tren', 'Kategori Kehadiran'
    ]
    fitur_cluster = [f for f in fitur_cluster if f in df.columns and f not in ['Status Kelulusan', 'IPS Semester Akhir']]

    X_cluster = df[fitur_cluster] 

    # Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Slider jumlah cluster
    n_clusters = st.slider("Jumlah Cluster", 2, 6, 3)

    # KMeans & Silhouette Score
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = cluster_labels
    sil_score = silhouette_score(X_scaled, cluster_labels)

    # Tampilkan jumlah anggota tiap cluster
    st.subheader("Jumlah Anggota Tiap Cluster")
    st.dataframe(df['Cluster'].value_counts().reset_index(name='Jumlah').rename(columns={'index': 'Cluster'}))

    # Tampilkan silhouette score
    st.info(f"Silhouette Score untuk {n_clusters} cluster: **{sil_score:.4f}**")

    # Visualisasi 2D (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Visualisasi Clustering Mahasiswa (PCA 2D)")
    st.pyplot(fig)

    # Visualisasi cluster pada fitur utama (IPK vs IPS Rata-rata) dengan centroid
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x='IPK', y='IPS Rata-rata',
        hue='Cluster',
        palette='Set1',
        data=df,
        alpha=0.7,
        ax=ax2
    )
    # Ambil centroid asli (bukan hasil PCA)
    centers = kmeans.cluster_centers_
    idx_ipk = fitur_cluster.index('IPK')
    idx_ips = fitur_cluster.index('IPS Rata-rata')
    # Inverse transform centroid ke skala asli
    centers_original = scaler.inverse_transform(centers)
    ax2.scatter(
        centers_original[:, idx_ipk],
        centers_original[:, idx_ips],
        s=200, c='yellow', marker='X', label='Centroid'
    )
    ax2.set_title('Clustering KMeans (IPK vs IPS Rata-rata)')
    ax2.set_xlabel('IPK')
    ax2.set_ylabel('IPS Rata-rata')
    ax2.set_xlim(1.5, 4.0)
    ax2.set_ylim(1.5, 4.0)
    ax2.legend()
    st.pyplot(fig2)

    # Tampilkan contoh data hasil cluster
    st.subheader("Contoh Data Hasil Clustering")
    st.dataframe(df[fitur_cluster + ['Cluster']].head(20))