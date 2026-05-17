python buat_data_provinsi.py   
python K-means_provinsi.py    
source .venv/bin/activate
------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ======================================================
# PROGRAM K-MEANS CLUSTERING PROVINSI INDONESIA
# Variabel:
# 1. IPM
# 2. Kemiskinan
# 3. TPT
# ======================================================

# 1. MEMBACA DATA
try:
    df = pd.read_csv("data_provinsi.csv")
    print("Berhasil memuat data_provinsi.csv.")
except FileNotFoundError:
    print("Error: File data_provinsi.csv tidak ditemukan.")
    print("Jalankan terlebih dahulu file buat_data_provinsi.py")
    exit()


# 2. MENAMPILKAN INFORMASI AWAL
print("\nInformasi Data:")
print(df.info())

print("\nLima Data Pertama:")
print(df.head())


# 3. MENYIAPKAN 3 VARIABEL UNTUK CLUSTERING
X = df[["IPM", "Kemiskinan", "TPT"]].values


# 4. STANDARISASI DATA
# Standarisasi perlu dilakukan karena skala IPM berbeda dengan Kemiskinan dan TPT.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. PROSES K-MEANS
# Jumlah cluster dibuat 3 agar mudah dianalisis:
# 1. Kesejahteraan relatif tinggi
# 2. Kesejahteraan relatif sedang
# 3. Kesejahteraan relatif rendah

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)


# 6. MENGAMBIL TITIK CENTROID
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)


# 7. MEMBERI LABEL CLUSTER
# Logika sederhana:
# IPM tinggi = semakin baik
# Kemiskinan rendah = semakin baik
# TPT rendah = semakin baik

rata_cluster = df.groupby("Cluster")[["IPM", "Kemiskinan", "TPT"]].mean()

rata_cluster["Skor_Kesejahteraan"] = (
    rata_cluster["IPM"]
    - rata_cluster["Kemiskinan"]
    - rata_cluster["TPT"]
)

urutan_cluster = rata_cluster["Skor_Kesejahteraan"].sort_values(ascending=False).index.tolist()

label_cluster = {
    urutan_cluster[0]: "Kesejahteraan Relatif Tinggi",
    urutan_cluster[1]: "Kesejahteraan Relatif Sedang",
    urutan_cluster[2]: "Kesejahteraan Relatif Rendah"
}

df["Kategori"] = df["Cluster"].map(label_cluster)


# 8. MENGHITUNG SILHOUETTE SCORE
silhouette = silhouette_score(X_scaled, df["Cluster"])


# 9. VISUALISASI 3D
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection="3d")

colors = {
    "Kesejahteraan Relatif Tinggi": "#2ecc71",
    "Kesejahteraan Relatif Sedang": "#f1c40f",
    "Kesejahteraan Relatif Rendah": "#e74c3c"
}

for kategori, warna in colors.items():
    subset = df[df["Kategori"] == kategori]

    ax.scatter(
        subset["IPM"],
        subset["Kemiskinan"],
        subset["TPT"],
        label=kategori,
        s=90,
        edgecolors="black",
        alpha=0.8,
        color=warna
    )

    # Menampilkan nama provinsi pada titik
    for _, row in subset.iterrows():
        ax.text(
            row["IPM"],
            row["Kemiskinan"],
            row["TPT"],
            row["Provinsi"],
            fontsize=7,
            alpha=0.7
        )

# Menampilkan centroid
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    marker="x",
    s=350,
    linewidths=5,
    color="black",
    label="Centroids"
)

ax.set_title("Hasil Clustering Provinsi di Indonesia Menggunakan K-Means")
ax.set_xlabel("IPM")
ax.set_ylabel("Kemiskinan (%)")
ax.set_zlabel("TPT (%)")
ax.legend()

plt.tight_layout()
plt.savefig("hasil_clustering_provinsi.png", dpi=300)
plt.show()


# 10. OUTPUT DI TERMINAL
print("\n==============================")
print("HASIL CLUSTERING K-MEANS")
print("==============================")

print("\nJumlah Data:", len(df))
print("Jumlah Cluster: 3")
print(f"Silhouette Score: {silhouette:.3f}")

print("\nTitik Tengah Centroid:")
for i, pt in enumerate(centroids):
    print(
        f"Centroid Cluster {i}: "
        f"IPM = {pt[0]:.2f}, "
        f"Kemiskinan = {pt[1]:.2f}%, "
        f"TPT = {pt[2]:.2f}%"
    )

print("\nRata-rata Setiap Cluster:")
print(df.groupby("Kategori")[["IPM", "Kemiskinan", "TPT"]].mean())

print("\nJumlah Provinsi per Cluster:")
print(df["Kategori"].value_counts())

print("\nHasil Clustering Lengkap:")
print(df[["Provinsi", "IPM", "Kemiskinan", "TPT", "Cluster", "Kategori"]])


# 11. MENYIMPAN HASIL CLUSTERING
df.to_csv("hasil_clustering_provinsi.csv", index=False)
print("\nFile hasil_clustering_provinsi.csv berhasil dibuat.")
print("File gambar hasil_clustering_provinsi.png juga berhasil dibuat.")